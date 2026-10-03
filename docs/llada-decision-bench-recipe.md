# Un décideur typé LLaDA sur decision-bench : 0.71–0.73 UNSEEN à ~50 ms

Recette complète et reproductible pour transformer un modèle de diffusion de langage masquée
(**LLaDA-8B** `GSAI-ML/LLaDA-8B-Instruct`, ou **iLLaDA-8B** `GSAI-ML/iLLaDA-8B-Instruct`) en un
**décideur typé** (choice / noul / score), lu au `[MASK]` en un seul forward.

Tout est mesuré sur GPU (SageMaker, A10G 24 Go). Aucun poids de base n'est modifié : les gains
viennent (1) du **canal de lecture** (readout) et (2) d'un **adapter LoRA** léger.

---

## Les chiffres honnêtes (à citer)

Bench complet `bench-v1` (2108 items notés), restreint aux sources **UNSEEN** : on retire les sources
vues à l'entraînement (`JUL_SEEN_EXTRA=dbpedia,mnli,banking77,agnews,trec,imdb,sst5,boolq,amazon,yelp`),
n = 1591. Config figée avant la mesure, même splitter (`scripts/score_split.py`) pour tous.

| système (UNSEEN, n=1591) | ALL [IC95] | choice | noul | score | p50 | GPU |
|---|---|---|---|---|---|---|
| Jev `jev-1.13.0` | 0.851 [0.833, 0.868] | 0.914 | 0.849 | 0.699 | — | API |
| wemm-4b v2.1 | 0.829 [0.810, 0.847] | 0.872 | 0.859 | 0.661 | — | 1 |
| **iLLaDA-8B + LoRA CE v7** | **0.733** [0.711, 0.754] | 0.806 | 0.762 | 0.493 | 54.5 ms | 1×A10G |
| LLaDA-8B + LoRA CE v7 | 0.713 [0.690, 0.734] | 0.797 | 0.745 | 0.437 | 51 ms | 1×A10G |

- iLLaDA − LLaDA (bootstrap apparié, mêmes items) : **+2.0 [−0.1, +4.1]**, sous le seuil de 3 points,
  **non significatif**. Avant adaptation l'écart était de +14.2 [+7.1, +21.3] (suite quick) : la même
  adaptation ramène les deux backbones au même niveau, le backbone n'est pas le facteur limitant.
- Écart à wemm : **−9.6 points UNSEEN**, dont **−16.8 sur score**.
- L'atout réel de la ligne LLaDA est la **latence** : ~50 ms plate, 1 GPU, un forward. Pas la qualité.

> Le **0.753** des sections 1 à 5 ci-dessous est un résultat **exploratoire** sur la suite *quick*
> (296 items), où les configs A–E et le routing `auto-ce` ont été conçus en regardant ces mêmes
> items : c'est un meilleur-de, pas un chiffre à citer. Le 0.889 de Jev est lui aussi sur *quick*
> (0.873 sur le bench complet).

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

## 1. La décomposition des effets (exploratoire, suite quick)

Matrice 2×2 sur `decision-bench` quick (296 items), même matériel (A10G), LLaDA-8B, ancien corpus
(~3 900 exemples). Les configs ont été choisies sur ces items : lire les **écarts**, pas les niveaux.

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

### Entraînement (recette v7, celle des chiffres honnêtes)
```bash
AWS_PROFILE=<profil> python deployment/sagemaker-eval/launch_train.py \
    --train /Users/jerome/dev/jul/data/mix/decision-v7.clean.jsonl \
    --base GSAI-ML/LLaDA-8B-Instruct \
    --stage a --loss ce \
    --lora-r 16 --epochs 2 --lr 5e-5 --lora-targets q_proj,k_proj,v_proj \
    --instance ml.g5.2xlarge
```
- Corpus **decision-v7** (suite Kev publique, 15 576 items, décontaminé du bench : 0 recouvrement
  textuel ; dbpedia et mnli sont partagés, d'où le split UNSEEN). Construit par
  `scripts/convert_decision_v7.py`. Le run exploratoire de la §1 utilisait ~3 900 exemples, r16,
  lr 1e-4, 1 epoch.
- `lora_alpha` vaut **2·r** par défaut (32 pour r=16). Un bug antérieur avait `alpha=256` avec r=16 :
  facteur 16× qui effondrait le readout. Corrigé ; ne pas fixer alpha à la main sans raison.
- **LoRA** sur les projections d'attention, backbone gelé. Le full fine-tuning d'un 8B s'effondre
  (oubli catastrophique) ; le LoRA préserve le modèle. ⚠️ LLaDA n'a pas de `o_proj` (sa projection de
  sortie s'appelle `attn_out`) : les suffixes par défaut `q/k/v/o` n'adaptent que **q/k/v** sur
  LLaDA, mais q/k/v/o sur iLLaDA. Pour comparer les deux, passer `--lora-targets q_proj,k_proj,v_proj`.
- **iLLaDA** : `--base GSAI-ML/iLLaDA-8B-Instruct`. Les launchers fixent seuls transformers 4.57.1 (version
  sauvegardée du remote code ; lm_head lié à l'embedding) et le `[MASK]` = id 5.
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
    --model llada-8b-instruct --suite full --readout auto-ce \
    --adapter s3://…/output/<ce-job>/output/model.tar.gz
JUL_SEEN_EXTRA='dbpedia,mnli,banking77,agnews,trec,imdb,sst5,boolq,amazon,yelp' \
    python scripts/score_split.py <decision-bench>/data/bench-v1.jsonl runs/<x>/predictions.jsonl
```
- Exploratoire (quick, ancien corpus) : ALL 0.753 (choice 0.805, noul 0.820, score 0.375), p50 51 ms.
- Honnête (full, UNSEEN, v7) : **0.713** (LLaDA-8B), **0.733** (iLLaDA-8B).

---

## 5. Recette condensée

1. **Backbone** : `GSAI-ML/iLLaDA-8B-Instruct` (ou LLaDA-8B), backend `llada` (diffusion masquée, lu
   au `[MASK]`). Shim transformers 5.x pour LLaDA, pin 4.57.1 pour iLLaDA.
2. **Readouts** (`lib/jul/mask.py`) : choice anchor + multi-token (routé à >10 options), noul oui/non
   multi-ancres, score ordinal par descriptions. Gratuits en latence.
3. **Adapter LoRA CE** : LoRA q/k/v, r16, alpha 32, lr 5e-5, 2 epochs, loss `ce`, sur decision-v7.
4. **Routing par type `auto-ce`** : choice→anchor (canal du CE), noul/score→readouts refaits.
   → **0.713 UNSEEN (LLaDA-8B), 0.733 (iLLaDA-8B)**.

Latence servie : ~51–55 ms/décision (A10G), plate et prévisible (un forward unique).

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
- **LLaDA2.0-mini** (MoE 16B, 1.4B actifs) : mesuré, **p50 458 ms / p95 637 ms sur 4×A10G** (`device_map
  auto`, 32.5 Go en BF16) contre 51 ms sur 1×A10G pour le dense. Critère d'arrêt atteint, non viable ici.
- **Tête de lecture apprise** (`lib/jul/llada_head.py`, `--head 1`) sur l'état caché au `[MASK]` :
  bootstrap apparié UNSEEN vs readout-logits, all Δ−0.5 (nul), **choice −2.7 [−5.4, 0.0]** (la clé est
  l'embedding de la lettre A/B/C, sans contenu d'option), score +5.6 non significatif (n=286). Confound :
  la LoRA est entraînée sur la loss de la tête, pas en CE.
- **Backbone plus fort à recette iso (iLLaDA-8B)** : +2.0 sur ALL, non significatif (voir en tête).
- **Tête choice listwise** (port de la lecture wemm, entraînée seule sur l'adapter iLLaDA-v7 figé) :
  choice UNSEEN **0.806 → 0.556, Δ−0.250 [−0.291, −0.208]** ; noul et score inchangés au bit près
  (`runs/illada-listwise/`). Rebrancher la lecture sans ré-entraîner l'adapter détruit choice.
- **Slot `[MASK]` par label** (choice) : effondrement du 1er slot (asymétrie de position).
- **Multi-token forcé sur le choice d'un modèle CE** : régresse (0.805 → 0.680) — d'où `auto-ce`.

---

## 7. Les écarts restants (UNSEEN, iLLaDA-8B v7)

| type | iLLaDA v7 | wemm v2.1 | écart | Jev | écart |
|---|---|---|---|---|---|
| choice | 0.806 | 0.872 | −6.6 | 0.914 | −10.8 |
| noul | 0.762 | 0.859 | −9.7 | 0.849 | −8.7 |
| **score** | 0.493 | 0.661 | **−16.8** | 0.699 | **−20.6** |
| ALL | 0.733 | 0.829 | −9.6 | 0.851 | −11.8 |

- **score** reste le plus gros fossé. Le backbone ne le ferme pas, et rebrancher la lecture de choice
  sur l'adapter figé le détruit. La cause ouverte est le corpus, en particulier la calibration ordinale.
  Piste non faite : un readout ordinal qui lit la position sur l'échelle plutôt que la description de
  chaque niveau.
- Le corpus de wemm v2.1 est fermé (open-weights, closed-data) : le test « même corpus » est impossible.

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

Résultats bruts : `runs/v7/` et `runs/illada-v7/` (chiffres honnêtes, bench complet) ;
`runs/dbench-E-CE-autoce.json` (le 0.753 exploratoire sur quick). Comparaisons appariées :
`scripts/paired_bootstrap.py --baseline … --candidate …`.
