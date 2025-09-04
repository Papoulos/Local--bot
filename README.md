# Passerelle API pour IA Locale

Ce projet fournit une passerelle d'API sécurisée et configurable pour interagir avec des modèles de langage locaux via [Ollama](https://ollama.ai/). Il agit comme un middleware, vous permettant de définir des "modèles" spécifiques dans un fichier de configuration qui peuvent pointer vers différents modèles Ollama ou même déclencher des chaînes complexes comme la Génération Augmentée par Récupération (RAG).

Toutes les appels API sont sécurisés par une clé API secrète. Les formats de requête et de réponse sont standardisés pour être compatibles avec l'API Chat Completions d'OpenAI, ce qui facilite l'intégration avec les outils existants.

---

## Objectif du projet

L'objectif principal de ce projet est de fournir une interface unifiée et sécurisée pour vos modèles de langage locaux. Il permet de centraliser l'accès, de gérer différentes configurations de modèles (par exemple, un modèle pour la discussion générale, un autre spécialisé pour l'analyse de documents) et de protéger l'accès avec une clé API.

## Features du projet

- **Routage de Modèles Configurable :** Définissez des mots-clés (ex: "chat", "cypher") dans un fichier `config.json` et mappez-les à différents modèles Ollama ou à des chaînes de traitement.
- **Authentification par Clé API :** Protégez vos modèles locaux avec une clé API secrète.
- **API Standardisée :** Formats de requête et de réponse compatibles avec OpenAI pour une intégration facile.
- **Support RAG :** Prise en charge intégrée de la Génération Augmentée par Récupération (RAG) avec `faiss` pour discuter avec vos documents.
- **Interface en Ligne de Commande (CLI) :** Une CLI simple pour démarrer le serveur et interagir avec les modèles.

## Prérequis

Assurez-vous d'avoir Python 3.8+ installé sur votre système.

## Installation du projet

Si vous lisez ceci, vous avez probablement déjà cloné le dépôt.

1.  **Installez les dépendances :**
    Ouvrez un terminal et exécutez la commande suivante pour installer les paquets Python nécessaires.
    ```bash
    pip install -r ollama_chat_rag/requirements.txt
    ```

2.  **Configurez l'environnement :**
    Copiez le fichier `.env.example` et modifiez-le pour définir votre clé API secrète.

    Sur Linux/macOS :
    ```bash
    cp ollama_chat_rag/.env.example ollama_chat_rag/.env
    ```
    Sur Windows :
    ```powershell
    copy ollama_chat_rag\.env.example ollama_chat_rag\.env
    ```
    Ensuite, ouvrez `ollama_chat_rag/.env` avec un éditeur de texte et changez la valeur de `API_KEY`.

## Utilisation des Scripts de Lancement

Pour simplifier le lancement, des scripts sont fournis pour chaque plateforme. Ils installent les dépendances (si nécessaire) puis démarrent le serveur avec les options que vous leur passez.

-   **Sur Linux ou macOS :**
    ```bash
    ./run.sh start --mock
    ```
-   **Sur Windows :**
    ```batch
    .\run.bat start --mock
    ```

Ces scripts sont le moyen le plus simple de démarrer.

## Configuration et Options

### 1. Configuration des Modèles

Ouvrez `ollama_chat_rag/config.json` pour définir vos modèles. Le `type` peut être `llm` pour un appel de modèle standard ou `rag` pour utiliser le système de recherche de documents.

```json
{
  "models": {
    "chat": {
      "type": "llm",
      "model_name": "mistral"
    },
    "cypher": {
      "type": "rag",
      "model_name": "mistral"
    }
  }
}
```

Pour les modèles de type `rag`, placez vos fichiers (`.txt`, etc.) dans le dossier `ollama_chat_rag/documents/`. Le système utilisera la bibliothèque `faiss` pour créer un index en mémoire de ces documents.

### 2. Options de Lancement du Serveur

Utilisez la CLI pour démarrer le serveur FastAPI manuellement.

```bash
python -m ollama_chat_rag.cli start
```

Voici les options disponibles :
-   `--mock` : Lance le serveur en mode "mock", sans connexion réelle à Ollama. Idéal pour les tests.
-   `--prod` : Lance le serveur en mode "production", en écoutant sur `127.0.0.1:8000` (plus sécurisé pour une exposition via tunnel). Par défaut, le serveur écoute sur `0.0.0.0:8000`.
-   `--background` ou `-b` : Lance le serveur en arrière-plan (non recommandé sur Windows).

**Exemple de lancement en mode mock :**
```bash
python -m ollama_chat_rag.cli start --mock
```

### 3. Commandes CLI Utilitaires

Des commandes sont aussi disponibles pour interroger directement votre API depuis le terminal (sans passer par `curl`) :
-   `python -m ollama_chat_rag.cli chat "Votre message ici"`
-   `python -m ollama_chat_rag.cli rag "Votre question sur les documents"`

## Exemples d'Utilisation de l'API

Vous pouvez interagir avec l'API en utilisant n'importe quel client HTTP, comme `curl`. Remplacez `VOTRE_CLE_API` par la clé définie dans le fichier `.env`.

### Exemple : Chat Standard (type `llm`)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "X-API-Key: VOTRE_CLE_API" \
-d '{
    "model": "chat",
    "messages": [
        {"role": "system", "content": "Vous êtes un assistant serviable."},
        {"role": "user", "content": "Bonjour, qui êtes-vous ?"}
    ]
}'
```

### Exemple : Chat RAG (type `rag`)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "X-API-Key: VOTRE_CLE_API" \
-d '{
    "model": "cypher",
    "messages": [
        {"role": "user", "content": "Quel est le sujet principal du document ?"}
    ]
}'
```

## Déploiement Sécurisé

Pour publier cette API de manière sécurisée sur internet, veuillez suivre les instructions détaillées dans le guide de déploiement :

**[➡️ Guide de Déploiement Sécurisé](./DEPLOYMENT.md)**
