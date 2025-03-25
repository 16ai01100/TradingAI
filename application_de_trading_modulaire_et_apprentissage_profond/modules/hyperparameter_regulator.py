import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
from collections import deque

class HyperparameterRegulator(nn.Module):
    """
    Système d'auto-régulation des hyperparamètres qui ajuste dynamiquement
    les paramètres du système en fonction des performances et des conditions du marché.
    """
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=8, history_size=50, 
                 learning_rate=0.001, adaptation_rate=0.05):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.history_size = history_size
        self.learning_rate = learning_rate
        self.adaptation_rate = adaptation_rate
        
        # Historique des performances et des hyperparamètres
        self.performance_history = deque(maxlen=history_size)
        self.hyperparam_history = deque(maxlen=history_size)
        self.market_state_history = deque(maxlen=history_size)
        
        # Réseau de prédiction des hyperparamètres optimaux
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Couche d'attention pour pondérer l'importance des différentes caractéristiques
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Réseau de prédiction des hyperparamètres
        self.hyperparam_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Normaliser les sorties entre 0 et 1
        )
        
        # Optimiseur
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Configuration du logging
        self.log_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'hyperparameter_regulator.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('HyperparameterRegulator')
        self.logger.info("Régulateur d'hyperparamètres initialisé")
    
    def store_experience(self, market_state, hyperparams, performance):
        """
        Stocke une expérience pour l'apprentissage futur.
        
        Args:
            market_state: État du marché
            hyperparams: Hyperparamètres utilisés
            performance: Performance obtenue
        """
        self.market_state_history.append(market_state)
        self.hyperparam_history.append(hyperparams)
        self.performance_history.append(performance)
    
    def learn(self):
        """
        Apprend à prédire les hyperparamètres optimaux en fonction des états du marché.
        
        Returns:
            float: Perte d'apprentissage
        """
        if len(self.performance_history) < 10:
            return None
        
        # Convertir les historiques en tenseurs
        market_states = torch.FloatTensor(np.array(list(self.market_state_history)))
        hyperparams = torch.FloatTensor(np.array(list(self.hyperparam_history)))
        performances = torch.FloatTensor(np.array(list(self.performance_history)))
        
        # Normaliser les performances
        performances = (performances - performances.mean()) / (performances.std() + 1e-8)
        
        # Encoder les états du marché
        encoded_states = self.encoder(market_states)
        
        # Calculer les poids d'attention
        attention_weights = F.softmax(self.attention(encoded_states).squeeze(-1), dim=0)
        
        # Appliquer l'attention
        weighted_encoding = torch.sum(encoded_states * attention_weights.unsqueeze(-1), dim=0)
        
        # Prédire les hyperparamètres optimaux
        predicted_hyperparams = self.hyperparam_predictor(weighted_encoding.unsqueeze(0))
        
        # Calculer la perte pondérée par les performances
        # Les hyperparamètres associés à de meilleures performances ont plus de poids
        performance_weights = F.softmax(performances, dim=0)
        weighted_loss = F.mse_loss(predicted_hyperparams.squeeze(0), 
                                  hyperparams, 
                                  reduction='none') * performance_weights.unsqueeze(-1)
        loss = weighted_loss.mean()
        
        # Optimisation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.logger.info(f"Apprentissage: perte={loss.item():.4f}")
        
        return loss.item()
    
    def predict_optimal_hyperparams(self, market_state):
        """
        Prédit les hyperparamètres optimaux pour un état de marché donné.
        
        Args:
            market_state: État du marché actuel
            
        Returns:
            dict: Hyperparamètres optimaux
        """
        # Convertir l'état du marché en tenseur
        market_state_tensor = torch.FloatTensor(market_state).unsqueeze(0)
        
        # Encoder l'état du marché
        encoded_state = self.encoder(market_state_tensor)
        
        # Prédire les hyperparamètres optimaux
        with torch.no_grad():
            predicted_hyperparams = self.hyperparam_predictor(encoded_state)
        
        # Convertir en dictionnaire
        hyperparam_names = [
            'learning_rate', 'batch_size_factor', 'dropout_rate', 'mutation_rate',
            'crossover_rate', 'elite_ratio', 'exploration_factor', 'memory_decay'
        ]
        
        # Ajuster les plages de valeurs pour chaque hyperparamètre
        hyperparams_dict = {
            'learning_rate': float(np.clip(predicted_hyperparams[0, 0].item() * 0.01, 0.0001, 0.01)),
            'batch_size': int(np.clip(predicted_hyperparams[0, 1].item() * 256, 16, 256)),
            'dropout_rate': float(np.clip(predicted_hyperparams[0, 2].item() * 0.5, 0.1, 0.5)),
            'mutation_rate': float(np.clip(predicted_hyperparams[0, 3].item() * 0.2, 0.001, 0.2)),
            'crossover_rate': float(np.clip(predicted_hyperparams[0, 4].item() * 0.5 + 0.5, 0.5, 0.95)),
            'elite_ratio': float(np.clip(predicted_hyperparams[0, 5].item() * 0.2, 0.05, 0.2)),
            'exploration_factor': float(np.clip(predicted_hyperparams[0, 6].item() * 2, 0.5, 2.0)),
            'memory_decay': float(np.clip(predicted_hyperparams[0, 7].item() * 0.5, 0.1, 0.5))
        }
        
        self.logger.info(f"Hyperparamètres prédits: {hyperparams_dict}")
        
        return hyperparams_dict
    
    def detect_market_regime_change(self, current_state, window_size=10, threshold=0.2):
        """
        Détecte les changements de régime de marché.
        
        Args:
            current_state: État actuel du marché
            window_size: Taille de la fenêtre d'observation
            threshold: Seuil de détection
            
        Returns:
            bool: True si un changement de régime est détecté
        """
        if len(self.market_state_history) < window_size:
            return False
        
        # Calculer la distance moyenne entre l'état actuel et les états récents
        recent_states = list(self.market_state_history)[-window_size:]
        current_state_tensor = np.array(current_state)
        
        distances = [np.linalg.norm(current_state_tensor - np.array(state)) for state in recent_states]
        avg_distance = np.mean(distances)
        
        # Calculer la volatilité des distances
        distance_std = np.std(distances)
        
        # Détecter un changement de régime si la distance moyenne dépasse le seuil
        regime_change = avg_distance > threshold * (1 + distance_std)
        
        if regime_change:
            self.logger.info(f"Changement de régime de marché détecté: distance={avg_distance:.4f}, seuil={threshold * (1 + distance_std):.4f}")
        
        return regime_change
    
    def adapt_hyperparams(self, current_hyperparams, optimal_hyperparams):
        """
        Adapte progressivement les hyperparamètres actuels vers les valeurs optimales.
        
        Args:
            current_hyperparams: Hyperparamètres actuels
            optimal_hyperparams: Hyperparamètres optimaux prédits
            
        Returns:
            dict: Hyperparamètres adaptés
        """
        adapted_hyperparams = {}
        
        for param_name, current_value in current_hyperparams.items():
            if param_name in optimal_hyperparams:
                # Adaptation progressive
                adapted_value = current_value + self.adaptation_rate * (optimal_hyperparams[param_name] - current_value)
                adapted_hyperparams[param_name] = adapted_value
            else:
                adapted_hyperparams[param_name] = current_value
        
        self.logger.info(f"Hyperparamètres adaptés: {adapted_hyperparams}")
        
        return adapted_hyperparams

class MarketRegimeDetector:
    """
    Détecteur de régimes de marché qui utilise des techniques de clustering
    et d'analyse de séries temporelles pour identifier les changements de régime.
    """
    def __init__(self, n_regimes=4, window_size=50, feature_dim=10):
        self.n_regimes = n_regimes
        self.window_size = window_size
        self.feature_dim = feature_dim
        
        # Historique des états du marché
        self.market_states = deque(maxlen=window_size*2)
        
        # Centroids des régimes (initialisés à None)
        self.regime_centroids = None
        
        # Régime actuel
        self.current_regime = None
        
        # Historique des régimes
        self.regime_history = deque(maxlen=window_size)
        
        # Configuration du logging
        self.log_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'market_regime_detector.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('MarketRegimeDetector')
        self.logger.info("Détecteur de régimes de marché initialisé")
    
    def add_market_state(self, market_state):
        """
        Ajoute un nouvel état du marché à l'historique.
        
        Args:
            market_state: État du marché
        """
        self.market_states.append(market_state)
    
    def _cluster_market_states(self):
        """
        Applique un algorithme de clustering pour identifier les régimes de marché.
        
        Returns:
            np.array: Centroids des régimes
        """
        if len(self.market_states) < self.n_regimes:
            return None
        
        # Convertir l'historique en tableau numpy
        states_array = np.array(list(self.market_states))
        
        # Initialiser les centroids aléatoirement
        indices = np.random.choice(len(states_array), self.n_regimes, replace=False)
        centroids = states_array[indices]
        
        # Algorithme K-means simplifié
        max_iterations = 100
        for _ in range(max_iterations):
            # Assigner chaque état au centroid le plus proche
            distances = np.array([[np.linalg.norm(state - centroid) for centroid in centroids] for state in states_array])
            labels = np.argmin(distances, axis=1)
            
            # Mettre à jour les centroids
            new_centroids = np.array([states_array[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else centroids[i] for i in range(self.n_regimes)])
            
            # Vérifier la convergence
            if np.allclose(centroids, new_centroids, rtol=1e-5):
                break
            
            centroids = new_centroids
        
        return centroids
    
    def identify_regime(self, market_state):
        """
        Identifie le régime de marché actuel.
        
        Args:
            market_state: État actuel du marché
            
        Returns:
            int: Indice du régime identifié
        """
        # Ajouter l'état à l'historique
        self.add_market_state(market_state)
        
        # Si pas assez de données ou pas de centroids, initialiser les centroids
        if self.regime_centroids is None or len(self.market_states) % 20 == 0:  # Recalculer périodiquement
            self.regime_centroids = self._cluster_market_states()
            
        if self.regime_centroids is None:
            return 0  # Régime par défaut
        
        # Calculer les distances aux centroids
        distances = [np.linalg.norm(np.array(market_state) - centroid) for centroid in self.regime_centroids]
        
        # Identifier le régime le plus proche
        regime = np.argmin(distances)
        
        # Enregistrer le régime dans l'historique
        self.regime_history.append(regime)
        self.current_regime = regime
        
        return regime
    
    def detect_regime_change(self, smoothing_window=5):
        """
        Détecte un changement de régime de marché.
        
        Args:
            smoothing_window: Fenêtre de lissage pour éviter les faux positifs
            
        Returns:
            bool: True si un changement de régime est détecté
        """
        if len(self.regime_history) < smoothing_window + 1:
            return False
        
        # Calculer le régime dominant récent
        recent_regimes = list(self.regime_history)[-smoothing_window:]
        regime_counts = np.bincount(recent_regimes)
        dominant_regime = np.argmax(regime_counts)
        
        # Calculer le régime dominant précédent
        previous_regimes = list(self.regime_history)[-(smoothing_window*2):-smoothing_window]
        if not previous_regimes:  # Si pas assez d'historique
            return False
            
        prev_regime_counts = np.bincount(previous_regimes)
        prev_dominant_regime = np.argmax(prev_regime_counts)
        
        # Détecter un changement de régime
        regime_change = dominant_regime != prev_dominant_regime
        
        if regime_change:
            self.logger.info(f"Changement de régime détecté: {prev_dominant_regime} -> {dominant_regime}")
        
        return regime_change
    
    def get_regime_characteristics(self):
        """
        Retourne les caractéristiques du régime actuel.
        
        Returns:
            dict: Caractéristiques du régime
        """
        if self.current_regime is None or self.regime_centroids is None:
            return {}
        
        # Extraire le centroid du régime actuel
        centroid = self.regime_centroids[self.current_regime]
        
        # Calculer la volatilité du régime
        regime_states = [np.array(state) for state, regime in zip(
            list(self.market_states)[-len(self.regime_history):], 
            self.regime_history) if regime == self.current_regime]
        
        if not regime_states:
            return {}
            
        regime_states = np.array(regime_states)
        volatility = np.mean(np.std(regime_states, axis=0))
        
        # Calculer la tendance du régime (simplifiée)
        trend = np.mean(centroid)
        
        # Calculer la stabilité du régime
        regime_durations = []
        current_regime = None
        current_duration = 0
        
        for regime in self.regime_history:
            if regime == current_regime:
                current_duration += 1
            else:
                if current_regime is not None:
                    regime_durations.append(current_duration)
                current_regime = regime
                current_duration = 1
                
        if current_duration > 0:
            regime_durations.append(current_duration)
            
        stability = np.mean(regime_durations) if regime_durations else 0
        
        return {
            'regime_id': self.current_regime,
            'volatility': float(volatility),
            'trend': float(trend),
            'stability': float(stability),
            'centroid': centroid.tolist()
        }