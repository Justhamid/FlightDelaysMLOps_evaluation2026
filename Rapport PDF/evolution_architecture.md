# Évolution : vers des batchs plus volumineux et un historique des prédictions

## Constat actuel

Aujourd'hui, le projet repose sur un simple CSV (200 000 lignes) versionné avec DVC, et l'API ne garde aucun historique des prédictions (juste des compteurs en mémoire, perdus à chaque redémarrage). Cette approche convient à l'échelle d'un TP, mais ne tiendrait pas si le client envoie des batchs beaucoup plus gros chaque jour et veut consulter l'historique de toutes les prédictions passées.

## Proposition : une architecture à trois niveaux

**1. Stockage des batchs bruts → Big Data (Data Lake)**
Quand les fichiers deviennent trop volumineux pour tenir en mémoire (plusieurs Go par batch), il faut un espace de stockage capable d'absorber de gros volumes bruts sans contrainte de structure, et des outils de traitement distribué (type Spark) pour les transformer par morceaux plutôt que tout charger d'un coup. C'est la seule option qui scale vraiment avec le volume.

**2. Données nettoyées pour l'entraînement → SQL**
Une fois les batchs nettoyés, les données de vol restent fondamentalement tabulaires (un vol = une ligne, des colonnes fixes). Une base SQL classique reste donc le bon choix ici : elle permet de faire des jointures et des agrégations facilement, ce qui est essentiel pour préparer les données d'entraînement et faire de l'analyse. Pas besoin de complexifier avec du NoSQL si la structure ne change pas.

**3. Historique des prédictions → NoSQL / base orientée temps**
Chaque prédiction de l'API (date, entrée, résultat, version du modèle) est un petit événement qu'on ajoute en continu, sans jamais le modifier. C'est un usage différent des données d'entraînement : beaucoup d'écritures, peu de mises à jour, et on veut souvent interroger "l'évolution dans le temps" (ex: taux de retard prédit par semaine). Une base orientée document ou série temporelle est plus adaptée à ce usage qu'une base SQL classique, et permet de construire un vrai tableau de bord de supervision qui survit aux redémarrages.

## Pourquoi cette combinaison plutôt qu'une seule techno

- **Volume** : le Big Data n'est utile que pour l'étape d'ingestion brute, à fort volume. Pas besoin de l'imposer partout.
- **Structure** : les données de vol restent structurées → le SQL garde tout son sens pour l'entraînement, inutile de le remplacer.
- **Requêtage** : SQL pour les analyses croisées (entraînement, BI), NoSQL/temps pour le suivi chronologique (supervision).
- **Traçabilité** : conserver l'historique des prédictions dans une vraie base (et plus en mémoire) permet de remonter sur n'importe quelle prédiction passée, avec quelle version du modèle elle a été faite.
- **Supervision** : une base persistante pour les prédictions permet de calculer des indicateurs de dérive sur de vraies périodes (jour, semaine, mois), au lieu de la fenêtre glissante actuelle qui se vide à chaque redémarrage de l'API.

## Limite assumée

Cette architecture n'a pas été implémentée dans ce projet (hors périmètre du TP) : c'est une proposition justifiée par les limites concrètes rencontrées (DVC pas adapté aux gros volumes, pas d'historique persistant des prédictions), pas une promesse de mise en œuvre.
