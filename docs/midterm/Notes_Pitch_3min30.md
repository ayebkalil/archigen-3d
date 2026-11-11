# Notes du pitch - ArchiGen-3D

Durée totale cible 3 min 30 s. Les temps sont des repères de répétition.

## Diapositive 1 - 30 secondes

Notre projet ArchiGen-3D transforme un plan 2D en une maison 3D interactive. Le studio aide à inspecter les pièces et à préparer les façades. Notre trinôme réunit Khalil Ayeb, Amine Ben Fadhel et Khawla Hawari. La capture montre une démonstration tracée. Pour mesurer la qualité des modèles, nous utilisons les jeux de validation et les tests documentés.

## Diapositive 2 - 45 secondes

Le pipeline relie les données aux deux modèles, puis à FastAPI et au viewer React et Three.js. Pour les plans, CubiCasa5K apporte quatre mille deux cents exemples d’entraînement, quatre cents de validation et quatre cents de test. Le modèle UNet++ partage un encodeur ResNet-34 et prédit la structure et les pièces. Pour les façades, Pix2Pix apprend sur les paires CMP. DVC versionne les données et les poids, et MLflow conserve les paramètres et les courbes. Nous travaillons sur une RTX 3050 de quatre gigaoctets.

## Diapositive 3 - 60 secondes

Les mesures montrent une progression nette après le recalage des annotations des plans. Le mIoU des pièces en validation passe de trente-sept pour cent à soixante pour cent, puis à soixante-deux virgule quatre avec le fine-tuning CAD. Plusieurs réglages changent entre les runs, donc nous ne leur attribuons pas un effet isolé. Pour les façades, le LPIPS du test historique diminue de zéro virgule soixante-trois à zéro virgule trente-et-un avec la version EMA. Les vingt-quatre runs sont conservés dans MLflow. Les derniers contrôles locaux donnent cent dix-neuf tests Python et vingt-sept tests frontend réussis.

## Diapositive 4 - 45 secondes

Notre premier verrou est la généralisation des plans. Sur douze plans externes, le champion commet en moyenne cinq virgule vingt-cinq erreurs de comptage des pièces. Le restylage aide certains cas, mais réduit les scores sémantiques en validation. Nous conservons donc le champion avec un fallback et prévoyons un nouveau test externe indépendant. Le second verrou est la qualité des façades sur quatre gigaoctets de VRAM. Le refiner 512 et la précision mixte permettent d’expérimenter localement. Les quatre styles architecturaux demandent encore des exemples étiquetés.

## Diapositive 5 - 30 secondes

Pour les quatre semaines restantes, nous proposons un premier jalon sur un nouveau test externe, piloté par Khalil. La deuxième semaine porte sur les modèles candidats et le parcours API et interface avec Amine. La troisième sécurise DVC, Docker, la CI et le monitoring avec Khawla et Amine. La dernière semaine rassemble les tests et la répétition. La répartition demande la validation du trinôme. Notre démonstration finale doit traiter un plan inédit et générer une façade avec une procédure reproductible.
