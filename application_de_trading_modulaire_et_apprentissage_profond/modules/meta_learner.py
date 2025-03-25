import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import logging
import os

class MetaLearner(nn.Module):
    """
    Système de méta-apprentissage qui permet au modèle d'apprendre à apprendre
    à partir de ses expériences passées et d'adapter ses stratégies de trading.
    """
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, memory_size=1000, batch_size=32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        # Mémoire d'expérience pour le méta-apprentissage
        self.experience_memory = deque(maxlen=memory_size)
        
        # Réseau d'encodage des expériences
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Réseau de méta-apprentissage
        self.meta_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Réseau de prédiction des hyperparamètres
        self.hyperparam_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 10)  # Prédiction de 10 hyperparamètres clés
        )
        
        # Optimiseur
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        
        # Configuration du logging
        self.log_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'meta_learner.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('MetaLearner')
        self.logger.info("Module de méta-apprentissage initialisé")
    
    def store_experience(self, state, action, reward, next_state, hyperparams, performance):
        """
        Stocke une expérience dans la mémoire pour l'apprentissage futur.
        
        Args:
            state: État du marché
            action: Action prise (stratégie sélectionnée)
            reward: Récompense obtenue
            next_state: État suivant du marché
            hyperparams: Hyperparamètres utilisés
            performance: Métriques de performance
        """
        self.experience_memory.append((state, action, reward, next_state, hyperparams, performance))
    
    def sample_batch(self):
        """
        Échantillonne un batch d'expériences pour l'apprentissage.
        
        Returns:
            tuple: Batch d'expériences
        """
        if len(self.experience_memory) < self.batch_size:
            return None
        
        batch = random.sample(self.experience_memory, self.batch_size)
        states, actions, rewards, next_states, hyperparams, performances = zip(*batch)
        
        # Conversion en tenseurs
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        hyperparams = torch.FloatTensor(np.array(hyperparams))
        performances = torch.FloatTensor(np.array(performances))
        
        return states, actions, rewards, next_states, hyperparams, performances
    
    def learn(self):
        """
        Effectue une étape d'apprentissage sur un batch d'expériences.
        
        Returns:
            float: Perte d'apprentissage
        """
        batch = self.sample_batch()
        if batch is None:
            return None
        
        states, actions, rewards, next_states, hyperparams, performances = batch
        
        # Encodage des états
        state_encodings = self.encoder(states)
        
        # Prédiction des hyperparamètres optimaux
        predicted_hyperparams = self.hyperparam_network(state_encodings)
        
        # Calcul de la perte sur les hyperparamètres
        hyperparam_loss = F.mse_loss(predicted_hyperparams, hyperparams)
        
        # Prédiction des représentations méta
        meta_outputs = self.meta_network(state_encodings)
        
        # Calcul de la perte de méta-apprentissage (corrélation avec les performances)
        performance_correlation = torch.sum(meta_outputs * performances.unsqueeze(1), dim=1)
        meta_loss = -torch.mean(performance_correlation)  # Maximiser la corrélation
        
        # Perte totale
        total_loss = hyperparam_loss + meta_loss
        
        # Optimisation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        self.logger.info(f"Méta-apprentissage: perte={total_loss.item():.4f}")
        
        return total_loss.item()
    
    def recommend_hyperparameters(self, state):
        """
        Recommande des hyperparamètres optimaux pour un état donné.
        
        Args:
            state: État du marché
            
        Returns:
            np.array: Hyperparamètres recommandés
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        state_encoding = self.encoder(state_tensor)
        hyperparams = self.hyperparam_network(state_encoding)
        
        return hyperparams.detach().numpy()[0]
    
    def get_meta_representation(self, state):
        """
        Obtient la représentation méta pour un état donné.
        
        Args:
            state: État du marché
            
        Returns:
            np.array: Représentation méta
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        state_encoding = self.encoder(state_tensor)
        meta_output = self.meta_network(state_encoding)
        
        return meta_output.detach().numpy()[0]
    
    def adapt_to_market(self, market_data, strategy_selector, ml_developer):
        """
        Adapte les modèles et stratégies en fonction des conditions de marché actuelles.
        
        Args:
            market_data: Données de marché récentes
            strategy_selector: Instance de StrategySelector
            ml_developer: Instance de MLDeveloper
            
        Returns:
            dict: Recommandations d'adaptation
        """
        # Extraire les caractéristiques pertinentes du marché
        market_features = self._extract_market_features(market_data)
        
        # Obtenir la représentation méta
        meta_representation = self.get_meta_representation(market_features)
        
        # Recommander des hyperparamètres
        recommended_hyperparams = self.recommend_hyperparameters(market_features)
        
        # Générer des recommandations d'adaptation
        recommendations = {
            'learning_rate': np.clip(recommended_hyperparams[0], 0.0001, 0.1),
            'batch_size': int(np.clip(recommended_hyperparams[1] * 100, 16, 256)),
            'dropout_rate': np.clip(recommended_hyperparams[2], 0.1, 0.5),
            'mutation_rate': np.clip(recommended_hyperparams[3], 0.001, 0.1),
            'market_condition': self._interpret_meta_representation(meta_representation),
            'strategy_weights': self._generate_strategy_weights(meta_representation)
        }
        
        self.logger.info(f"Recommandations d'adaptation générées: {recommendations}")
        
        # Appliquer les recommandations si possible
        if strategy_selector:
            strategy_selector.custom_weights = recommendations['strategy_weights']
        
        if ml_developer:
            ml_developer.update_hyperparameters(recommendations)
        
        return recommendations
    
    def _extract_market_features(self, market_data):
        """
        Extrait les caractéristiques pertinentes du marché.
        
        Args:
            market_data: Données de marché récentes
            
        Returns:
            np.array: Caractéristiques du marché
        """
        # Calculer les rendements
        returns = market_data['close'].pct_change().dropna().values
        
        # Calculer la volatilité (écart-type des rendements)
        volatility = np.std(returns)
        
        # Calculer la tendance (moyenne des rendements)
        trend = np.mean(returns)
        
        # Calculer le volume moyen
        volume = market_data['volume'].mean() if 'volume' in market_data else 0
        
        # Calculer des indicateurs techniques de base
        # RSI (Relative Strength Index)
        delta = market_data['close'].diff().dropna()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs.iloc[-1])) if loss.iloc[-1] != 0 else 50
        
        # Moyennes mobiles
        ma_short = market_data['close'].rolling(window=10).mean().iloc[-1]
        ma_long = market_data['close'].rolling(window=30).mean().iloc[-1]
        ma_ratio = ma_short / ma_long if ma_long != 0 else 1
        
        # Créer un vecteur de caractéristiques
        features = np.array([
            trend,
            volatility,
            volume,
            rsi / 100,  # Normaliser entre 0 et 1
            ma_ratio - 1,  # Centrer autour de 0
            np.mean(returns[-5:]),  # Tendance récente
            np.std(returns[-5:]),   # Volatilité récente
            np.percentile(returns, 10),  # Risque de perte
            np.percentile(returns, 90),  # Potentiel de gain
            np.mean(market_data['volume'].pct_change().dropna().values) if 'volume' in market_data else 0  # Tendance du volume
        ])
        
        return features
    
    def _interpret_meta_representation(self, meta_representation):
        """
        Interprète la représentation méta pour déterminer la condition du marché.
        
        Args:
            meta_representation: Représentation méta du marché
            
        Returns:
            str: Condition du marché
        """
        # Calculer la moyenne des composantes
        mean_value = np.mean(meta_representation)
        
        # Calculer la variance des composantes
        variance = np.var(meta_representation)
        
        # Déterminer la condition du marché en fonction de la moyenne et de la variance
        if variance > 0.2:  # Haute variance indique un marché volatile
            condition = 'volatile'
        elif mean_value > 0.6:  # Valeur moyenne élevée indique un marché haussier
            condition = 'bullish'
        elif mean_value < 0.4:  # Valeur moyenne faible indique un marché baissier
            condition = 'bearish'
        else:  # Valeur moyenne intermédiaire indique un marché latéral
            condition = 'sideways'
        
        return condition
    
    def _generate_strategy_weights(self, meta_representation):
        """
        Génère des poids pour différentes stratégies en fonction de la représentation méta.
        
        Args:
            meta_representation: Représentation méta du marché
            
        Returns:
            dict: Poids des stratégies
        """
        # Utiliser différentes parties de la représentation méta pour pondérer les stratégies
        # Diviser la représentation en groupes pour différentes stratégies
        n_strategies = 5  # Nombre de stratégies à pondérer
        group_size = len(meta_representation) // n_strategies
        
        # Calculer les poids bruts en prenant la moyenne de chaque groupe
        raw_weights = [np.mean(meta_representation[i*group_size:(i+1)*group_size]) for i in range(n_strategies)]
        
        # Normaliser les poids pour qu'ils somment à 1
        total_weight = sum(raw_weights)
        normalized_weights = [w / total_weight for w in raw_weights] if total_weight > 0 else [1/n_strategies] * n_strategies
        
        # Associer les poids aux noms des stratégies
        strategy_names = ['trend_following', 'mean_reversion', 'momentum', 'breakout', 'volatility_based']
        
        return dict(zip(strategy_names, normalized_weights))