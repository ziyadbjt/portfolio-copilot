# Copilote d'investissement agentique : gestion de portefeuille et risk management

## Contexte

Projet perso de Ziyad (futur MSc Finance & Big Data à NEOMA), destiné à son CV et à son GitHub. L'outil sert aussi à gérer son propre portefeuille d'actions. Il doit être démontrable en entretien : chaque choix technique doit pouvoir être expliqué.

Mots d'ordre :
- les calculs de risque sont faits par du code déterministe (jamais par un LLM)
- les agents proposent, l'humain valide
- aucun ordre n'est jamais exécuté par l'outil (lecture seule)

## Politique d'investissement (source de vérité, à mettre dans config/policy.yaml)

- Devise de base : EUR (les titres US portent le risque EUR/USD)
- Point de départ : aucun titre détenu, l'outil doit d'abord sélectionner les candidats (screening)
- Horizon : 5 ans
- Perte maximale tolérée : 20 % mesurée en max drawdown depuis le plus haut du portefeuille
- Univers : actions Europe + US
- Style du cœur : dividendes (entreprises stables, payout soutenable)
- Exclusions sectorielles : tabac, alcool, jeux d'argent
- Structure : poche défensive + poche actions (cœur / satellite)
  - poche défensive : fonds monétaire EUR (XEON.DE). Décidé à l'étape 4 : 100 % actions
    donne -51 % en 2008 et -42 % en 2020, incompatible avec la limite de 20 %.
    Le risque se pilote donc par la part d'actions, choisie par l'optimiseur.
  - poche actions : 80 % cœur / 20 % satellite (paramétrable)
  - satellite : thématiques IA et énergie (exemples : Micron, Intel, TotalEnergies ; SK Hynix écartée car cotée en Corée, hors univers)
- Limites à définir dans la config : poids max par ligne, par secteur, par pays, part max du satellite
- Risque de change EUR/USD à suivre

## Environnement

- Python via Anaconda (environnement conda dédié au projet)
- Éditeur : VS Code, cellules `# %%` utilisables pour exécuter par blocs
- Pas d'interface web : sorties en graphiques matplotlib, rapport HTML ou PDF généré automatiquement, export Excel
- Librairies : pandas, numpy, scipy, matplotlib, yfinance, PyPortfolioOpt (choisie pour sa lisibilité)
- Env conda : `portfolio-copilot` (installé avec /opt/anaconda3), défini dans environment.yml
- La clé API Anthropic (phase 2) va dans un fichier .env, à ajouter au .gitignore. Ne jamais la committer.

## Architecture visée

```
config/policy.yaml        politique d'investissement
config/universe.csv       liste des titres candidats (ticker, nom, pays, secteur, poche cœur/satellite)
src/data.py               récupération et nettoyage des prix, dividendes
src/screener.py           filtres et classement des candidats (exclusions, dividendes, payout, volatilité)
src/risk.py               volatilité, corrélations, VaR, CVaR, max drawdown, concentration, risque de change, stress tests, Monte Carlo
src/optimizer.py          optimisation cœur/satellite sous contraintes
src/policy_check.py       vérification des limites de la politique
src/report.py             graphiques et rapport HTML ou PDF
tests/                    tests unitaires sur les calculs de risque
main.py                   enchaîne le tout (phase 1)
```

Règle de conception : chaque fonction fait une seule chose, prend des entrées claires et renvoie un résultat clair. Les fonctions de calcul deviendront des outils pour l'agent en phase 2 sans être réécrites.

## Mesures de risque et règle de décision

- Max drawdown historique du portefeuille proposé
- Stress tests : 2008, mars 2020, 2022
- CVaR (Expected Shortfall) en plus de la VaR
- Monte Carlo sur 5 ans : probabilité de dépasser un drawdown de 20 %
- Concentration par ligne, secteur, pays
- Contribution au risque de chaque ligne

Une allocation est acceptée seulement si le drawdown en stress test reste sous 20 % et si la probabilité de dépasser 20 % en Monte Carlo reste sous un seuil configurable (10 % par défaut). Sinon, l'optimiseur augmente la poche défensive (et peut réduire le satellite). Ces mesures estiment un risque à partir du passé, elles ne garantissent rien : à écrire clairement dans le README.

## Plan de travail

### Phase 1 : script sans LLM (MVP)
1. Config de politique et chargement
2. Données de prix (yfinance) avec cache local
3. Screening : filtres déterministes et classement des candidats, l'humain valide la short-list
4. Module de risque, testé seul sur des cas connus
5. Optimisation sous contraintes
6. main.py qui enchaîne et produit un rapport avec graphiques
7. Explication générée par un modèle de texte (gabarit Jinja, pas de LLM) rempli avec les chiffres
8. README propre avec limites et hypothèses

### Phase 2 : couche agent (API Claude)
1. Transformer les fonctions de calcul en outils (tool use)
2. Agent orchestrateur qui appelle les outils, corrige une allocation hors limites et rédige l'explication
3. Garde-fous : outils en lecture seule, validation humaine, logs complets de chaque décision
4. Tests de robustesse : prompt injection via les news, hallucinations sur les chiffres (les chiffres viennent toujours des outils)
5. Cartographie AI Act : transparence, documentation, supervision humaine

### Phase 3 : extensions
- RAG sur rapports annuels et news (base vectorielle, citations des sources)
- Serveurs MCP pour les sources de données
- Backtest correct puis paper trading de 1 à 2 mois minimum avant tout usage réel

## Règles de travail

- Ne dire "agentique" dans le README et sur le CV qu'à partir de la phase 2 terminée.
- Expliquer chaque module après l'avoir écrit : Ziyad doit pouvoir justifier chaque choix en entretien (pourquoi la CVaR plutôt que la VaR, comment le périmètre de l'agent est sécurisé).
- Tester les fonctions de risque sur des données simulées dont on connaît le résultat avant d'utiliser les vraies données.
- Commits petits et clairs, un par étape.
- Répondre en français, ton direct et simple.
