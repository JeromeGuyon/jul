# De zéro à 0.753 : un décideur typé LLaDA sur decision-bench

Recette complète et reproductible pour transformer **LLaDA-8B** (un modèle de diffusion de langage
masquée, poids publics `GSAI-ML/LLaDA-8B-Instruct`) en un **décideur typé** (choice / noul / score)
qui atteint **0.753** de précision sur `decision-bench` (suite *quick*, 296 items), contre **0.517**
au départ et **0.889** pour Jev (la référence).

Tout est mesuré sur GPU (SageMaker, A10G 24 Go). Aucun poids de base n'est modifié : les gains
viennent (1) du **canal de lecture** (readout) et (2) d'un **adapter LoRA** léger. Chaque effet a été
isolé par une matrice 2×2.

---

## 0. Ce qu'il faut comprendre d'abord

LLaDA ne génère pas de texte de gauche à droite. Il prédit les tokens **masqués** (`[MASK]`) en un
seul passage bidirectionnel. Pour en faire un décideur, on place un `[MASK]` là où va la réponse, on
fait **un forward**, et on lit la distribution de probabilité au masque. La façon de lire cette
distribution — le **readout** — pèse autant que le modèle lui-même.

Trois types de questions (l'API imite celle de Jev) :
- **choice** : choisir une option parmi N.
- **noul** : oui / non (probabilité de « oui »).
- **score** : noter sur une échelle ordonnée.

---

## 1. Le résultat, et la décomposition des effets

Matrice 2×2 sur `decision-bench` quick (296 items), même matériel (A10G) :

| Config | ALL | choice | noul | score | latence p50 |
|--------|:---:|:------:|:----:|:-----:|:-----------:|
| **A** — brut + readout d'origine | 0.517 | 0.664 | 0.461 | 0.225 | 48 ms |
| **B** — brut + readouts refaits | 0.679 | 0.664 | 0.781 | 0.400 | 47 ms |
| **C** — LoRA CE + readout d'origine | 0.659 | 0.805 | 0.648 | 0.225 | 53 ms |
| **D** — LoRA CE + readouts refaits (auto) | 0.699 | 0.680 | 0.820 | 0.375 | 51 ms |
| **E** — LoRA CE + readout par type (`auto-ce`) | **0.753** | **0.805** | **0.820** | 0.375 | 51 ms |
| Jev (référence) | 0.889 | 0.93 | 0.92 | 0.65 | — |

Effets isolés :
- **Readouts seuls (B − A) = +0.162**, quasi gratuit (même latence). Débloque noul (0.46 → 0.78) et
  score (0.23 → 0.40).
- **Entraînement LoRA CE seul (C − A) = +0.142**. Aide surtout le choice (0.66 → 0.805) et le noul.
- **Les deux ne s'additionnent pas naïvement** : ils réparent en partie le même trou (noul). Il faut
  un **routing par type** pour cumuler sans conflit (voir §4).

**Coût du meilleur (E) : +0.236 de qualité vs le départ, pour +3 ms de latence.**

---

## 2. Étape 1 — Le readout, le levier gratuit (0.517 → 0.679)

Aucun entraînement. On change seulement *comment on lit* la sortie du modèle. Trois readouts, un par
type. Ils vivent dans `lib/jul/mask.py` (classe `MaskReader`), activés par la variable d'environnement
`JUL_LLADA_READOUT`.

### 2a. `choice` — readout par ancre + multi-token (le débloqueur des grandes cardinalités)

- **≤ 10 options** : *anchor*. On compare le premier token de chaque option (une lettre/clé) lu au
  `[MASK]`. Net et rapide.
- **> 10 options** : *multi-token, sequence-likelihood*. Le readout par ancre **sature** au-delà de
  ~26 classes (Banking77, 72 labels : **0.02**, quasi zéro — un problème de *canal*, pas de poids).
  Solution :
  1. tokeniser le **texte complet** de chaque label ;
  2. retirer le **préfixe commun** à tous les labels (le boilerplate porte 0 signal) ;
  3. poser un bloc de `K = max(longueur des suffixes)` `[MASK]` ;
  4. **un seul forward** ; lire le log-softmax à chaque position ;
  5. score du label = **moyenne** (length-normalized, *pas* la somme) des log-prob de ses propres
     tokens ; argmax.
  - Résultat : Banking77 **0.02 → 0.44**.
- Le routing **≤10 / >10** est le mode `auto`.

Piège vérifié (arXiv:2608.14649) : **ne jamais** donner à chaque label son propre slot `[MASK]` dans
une même séquence (`label1: [MASK]; label2: [MASK]…`). Le premier slot s'effondre (asymétrie de
position). Toujours interroger à la **même** position (bloc unique ci-dessus).

### 2b. `noul` — readout oui/non naturel (le plus gros gain, 0.46 → 0.78)

Le readout d'origine comparait le 1er token des mots « false »/« true » listés comme options, avec un
prompt bancal (liste + hint + `Answer:`) : **0.461**, quasi le hasard. Le canal était mauvais.

Le nouveau (`_noul_scores`) :
1. prompt naturel : `<état>\n<question>\nAnswer (yes or no):` puis `[MASK]` ;
2. lire les logits au masque sur **plusieurs paires d'ancres** : `(yes,no)`, `(Yes,No)`,
   `(true,false)`, `(True,False)` ;
3. moyenner **en log** les « oui » d'un côté, les « non » de l'autre ;
4. `P(oui) = softmax([oui, non])`.
- Résultat : noul **0.461 → 0.781**.

### 2c. `score` — readout ordinal (0.23 → 0.40)

Lire les chiffres « 0,1,2… » comme ancres capte mal l'ordre. On lit à la place la **description texte
de chaque niveau** par sequence-likelihood (même mécanisme que le choice multi-token), puis le client
calcule l'espérance du niveau `E[level]`.
- Résultat : score **0.225 → 0.400**.

> Ces trois readouts se lisent dans **le même forward unique** : gain de qualité **sans coût de
> latence** (47 ms vs 48 ms).

---

## 3. Étape 2 — L'adapter LoRA CE (aide surtout le choice, 0.66 → 0.805)

Un fine-tuning **léger** (LoRA) du modèle sur des décisions typées, avec une **cross-entropy simple**.

### Données
Corpus de décisions typées `state / question / options / gold` (ici : ~3 900 exemples issus du mix
de familles, avec soft labels d'un modèle-professeur possibles mais **la CE simple sur le gold suffit
et généralise mieux** — mesuré : la CE plate bat les variantes pondérées sur le zero-shot).

### Entraînement (SageMaker, ~2 100 s sur A10G spot)
```bash
AWS_PROFILE=<profil> python deployment/sagemaker-eval/launch_train.py \
    --train <corpus.soft.jsonl> \
    --base GSAI-ML/LLaDA-8B-Instruct \
    --stage a --loss ce \
    --lora-r 16 --lr 1e-4 --epochs 1 \
    --instance ml.g5.2xlarge --spot
```
- **LoRA** sur les projections d'attention (`q/k/v/o`), backbone gelé. Le full fine-tuning d'un 8B
  s'effondre (oubli catastrophique) ; le LoRA préserve le modèle.
- La loss `ce` (gold dur) est le bon défaut. `gift` (pondération par entropie, arXiv:2509.20863) et
  `diffusion` (facteur 1/t) sont implémentées mais **régressent le zero-shot** ici (sur-spécialisent
  les familles difficiles, oublient les fortes).
- Le modèle apprend à lire le choice via le **canal ancre** — retenir ce point pour l'étape 3.

L'adapter sort dans `s3://…/output/<job>/output/model.tar.gz`.

> Note technique : le remote code de LLaDA date de transformers ~4.4x. Sous transformers 5.x, un shim
> est nécessaire (`lib/jul/backends/llada.py` : `all_tied_weights_keys`, `tie_weights`, `use_cache`).
> Le MoE `LLaDA-MoE-7B-A1B` exige transformers **4.53.3** (pin conditionnel) et un LoRA *routing-guided*
> (MoE-Sieve) — mais il n'a pas fait mieux que le dense et est ~8× plus lent : **rester sur le dense**.

---

## 4. Étape 3 — Le routing par type `auto-ce` (le liant, → 0.753)

Découverte clé : **un conflit de canal sur le choice.** Le modèle CE a appris à lire le choice via
l'ancre. Si on le lit ensuite en multi-token (mode `auto`), le choice **régresse** (0.805 → 0.680).

`auto-ce` prend donc **le meilleur canal pour chaque type** :

| type | canal | pourquoi |
|------|-------|----------|
| choice | **anchor** (toujours, jamais multi-token) | le CE a appris ce canal → 0.805 |
| noul | **readout oui/non refait** | 0.820 |
| score | **readout ordinal refait** | 0.375 |

```bash
AWS_PROFILE=<profil> python deployment/sagemaker-eval/launch_dbench.py \
    --model llada-8b-instruct --suite quick --readout auto-ce \
    --adapter s3://…/output/<ce-job>/output/model.tar.gz
```
- Résultat : **ALL 0.753** (choice 0.805, noul 0.820, score 0.375), latence p50 51 ms.

---

## 5. Recette condensée (de 0 à 0.753)

1. **Backbone** : `GSAI-ML/LLaDA-8B-Instruct`, backend `llada` (diffusion masquée, lu au `[MASK]`),
   shim transformers 5.x.
2. **Readouts** (`lib/jul/mask.py`) : choice anchor + multi-token (routé à >10 options), noul oui/non
   multi-ancres, score ordinal par descriptions. → **0.679, gratuit.**
3. **Adapter LoRA CE** : LoRA `q/k/v/o`, loss `ce`, 1 epoch, sur ~4 k décisions typées. → aide le
   choice (0.805) et le noul.
4. **Routing par type `auto-ce`** : choice→anchor (canal du CE), noul/score→readouts refaits, pour
   cumuler sans le conflit de canal. → **0.753.**

Budget : 1 job d'entraînement (~35 min GPU) + 1 job d'éval (~15 s de calcul, hors provisioning).
Latence servie : ~51 ms/décision (A10G), plate et prévisible (un forward unique).

---

## 6. Ce qui NE marche pas (négatifs mesurés, à ne pas réexplorer)

- **Astuces d'inférence sur un modèle entraîné** (multi-mask `n_mask>1`, demasking itératif
  `n_steps>1`) : aident le zero-shot brut mais **dégradent** un modèle CE (elles bruitent ce que le
  LoRA a appris) ; +latence.
- **Loss GIFT / diffusion pour cet objectif** : régresse le zero-shot vs la CE simple (oubli
  catastrophique des familles fortes).
- **LLaDA-MoE-7B-A1B** : chargeable (pin 4.53.3 + routing-guided LoRA), **pas meilleur** que le dense,
  et **~8× plus lent** (813 ms vs 51 ms) — **en BF16** (`JUL_DTYPE=bfloat16`, pas de 4-bit). ⚠️ Ce 813 ms
  mesure l'**implémentation MoE naïve de transformers** (experts exécutés en boucle Python), **pas**
  l'architecture MoE en soi. Il **ne se transpose pas** à d'autres MoE : `LLaDA2.0-mini` (16B total,
  1.4B actifs, base Ling 2.0) est un modèle différent dont la latence doit être **mesurée séparément**
  avant toute conclusion.
- **Slot `[MASK]` par label** (choice) : effondrement du 1er slot (asymétrie de position).
- **Multi-token forcé sur le choice d'un modèle CE** : régresse (0.805 → 0.680) — d'où `auto-ce`.

---

## 7. Le seul gros écart restant à Jev

- choice 0.805 vs 0.93, noul 0.820 vs 0.92 : proches.
- **score 0.375 vs 0.65** : le dernier vrai fossé. Le readout ordinal reste intrinsèquement difficile
  au `[MASK]`. Pistes non faites : readout ordinal dédié (lire la *position* sur l'échelle, pas la
  description de chaque niveau), ou calibration few-shot (biais par niveau sur ~200 exemples de val —
  hors du zero-shot strict).
- Global : **0.753 vs 0.889.** Parti de 0.517, on a fermé ~55 % de l'écart, essentiellement par le
  **canal de lecture** (gratuit) + un **LoRA CE** léger, sans toucher aux poids de base.

---

## 8. Fichiers de la recette (dans `jul-bis`)

| Fichier | Rôle |
|---------|------|
| `lib/jul/backends/llada.py` | backend diffusion masquée, lecture au `[MASK]`, shim transformers 5.x |
| `lib/jul/mask.py` | les readouts : `_read_option_logits` (anchor), `_multitoken_scores`, `_noul_scores`, routing `auto` / `auto-ce` |
| `scripts/llada_train.py` | pertes (ce / diffusion / gift) + `build_example` + `train_step` |
| `deployment/sagemaker-eval/launch_train.py` + `train_entry.py` | entraînement LoRA sur SageMaker |
| `deployment/sagemaker-eval/launch_dbench.py` + `dbench_entry.py` | run decision-bench sur SageMaker, sortie au format du bench |
| `runs/dbench-{A..E}*.json` | les rapports de la matrice 2×2 |

Résultats bruts : `runs/dbench-E-CE-autoce.json` (le 0.753).
