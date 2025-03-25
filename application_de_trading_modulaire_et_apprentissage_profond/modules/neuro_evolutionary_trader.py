import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt

from .morphogenetic_optimizer import MorphogeneticOptimizer, SignalFusion, NTMController
from .evolutionary_population import EvolutionaryPopulation
from .meta_learner import MetaLearner

class NeuroEvolutionaryTrader:
    """
    Système de trading avancé qui combine l'optimisation morphogénétique, l'évolution
    génétique et le méta-apprentissage pour s'adapter dynamiquement aux conditions du marché.
    """
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=1, population_size=20,
                 memory_size=256, memory_dim=32, meta_memory_size=1000):
        """
        Initialise le système de trading neuro-évolutionnaire.
        
        Args:
            input_dim (int): Dimension des données d'entrée
            hidden_dim (int): Dimension des couches cachées
            output_dim (int): Dimension de sortie
            population_size (int): Taille de la population évolutionnaire
            memory_size (int): Taille de la mémoire externe
            memory_dim (int): Dimension de la mémoire externe
            meta_memory_size (int): Taille de la mémoire pour le méta-apprentissage
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialisation du génome template
        self.genome_template = self._create_genome_template()
        
        # Initialisation de la population évolutionnaire
        self.population = EvolutionaryPopulation(
            population_size=population_size,
            genome_template=self.genome_template,
            genome_size=8,
            gene_length=8
        )
        
        # Initialisation du contrôleur de mémoire
        self.memory_controller = NTMController(
            mem_size=memory_size,
            mem_dim=memory_dim,
            num_read_heads=2,
            num_write_heads=2,
            forget_factor=0.1
        )
        
        # Initialisation du méta-apprenant
        self.meta_learner = MetaLearner(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            memory_size=meta_memory_size
        )
        
        # Fusion des signaux
        self.signal_fusion = SignalFusion(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=4
        )
        
        # Meilleur réseau actuel
        self.best_network = None
        self.best_fitness = -float('inf')
        
        # Historique des performances
        self.performance_history = []
        
        # Configuration du logging
        self.log_dir = os.path.join(os.getcwd(), 'logs')
        self.model_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'neuro_evolutionary_trader.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('NeuroEvolutionaryTrader')
        self.logger.info("Système de trading neuro-évolutionnaire initialisé")
    
    def _create_genome_template(self):
        """
        Crée un génome template pour initialiser la population.
        
        Returns:
            list: Génome template
        """
        # Créer un génome de base avec une architecture raisonnable
        genome = [
            # Gène 1: Couche d'entrée -> Couche cachée 1
            # [activer?, taille_entrée, taille_sortie, activer_activation?, type_activation, activer_norm?, activer_dropout?, taux_dropout]
            np.array([1.0, 0.1, 0.64, 0.8, 0.6, 0.7, 0.6, 0.2]),  # Couche linéaire avec ReLU, BatchNorm et Dropout
            
            # Gène 2: Couche cachée 1 -> Couche cachée 2
            np.array([1.0, 0.64, 0.32, 0.8, 0.6, 0.7, 0.6, 0.2]),  # Couche linéaire avec ReLU, BatchNorm et Dropout
            
            # Gène 3: Couche cachée 2 -> Couche de sortie
            np.array([1.0, 0.32, 0.01, 0.6, 0.4, 0.0, 0.0, 0.0]),  # Couche linéaire avec activation
            
            # Gènes supplémentaires pour permettre l'évolution d'architectures plus complexes
            np.array([0.4, 0.32, 0.16, 0.8, 0.6, 0.7, 0.6, 0.2]),  # Désactivé par défaut
            np.array([0.4, 0.16, 0.08, 0.8, 0.6, 0.7, 0.6, 0.2]),  # Désactivé par défaut
            np.array([0.4, 0.08, 0.04, 0.8, 0.6, 0.7, 0.6, 0.2]),  # Désactivé par défaut
            np.array([0.4, 0.04, 0.02, 0.8, 0.6, 0.7, 0.6, 0.2]),  # Désactivé par défaut
            np.array([0.4, 0.02, 0.01, 0.8, 0.6, 0.0, 0.0, 0.0]),  # Désactivé par défaut
        ]
        
        return genome
    
    def preprocess_data(self, data):
        """
        Prétraite les données de marché pour l'apprentissage.
        
        Args:
            data (pd.DataFrame): Données de marché brutes
            
        Returns:
            tuple: (données_prétraitées, cibles)
        """
        # Calculer les rendements
        data['returns'] = data['close'].pct_change()
        
        # Calculer des indicateurs techniques
        # RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Moyennes mobiles
        data['sma_10'] = data['close'].rolling(window=10).mean()
        data['sma_30'] = data['close'].rolling(window=30).mean()
        data['sma_ratio'] = data['sma_10'] / data['sma_30']
        
        # MACD
        data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
        data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        data['bb_std'] = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + 2 * data['bb_std']
        data['bb_lower'] = data['bb_middle'] - 2 * data['bb_std']
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        # Volatilité
        data['volatility'] = data['returns'].rolling(window=20).std()
        
        # Supprimer les lignes avec des valeurs manquantes
        data = data.dropna()
        
        # Sélectionner les caractéristiques et la cible
        features = [
            'returns', 'rsi', 'sma_ratio', 'macd', 'macd_hist',
            'bb_width', 'volatility'
        ]
        
        # Normaliser les caractéristiques
        for feature in features:
            mean = data[feature].mean()
            std = data[feature].std()
            data[feature] = (data[feature] - mean) / (std if std > 0 else 1)
        
        # Préparer les données d'entrée et les cibles
        X = data[features].values
        y = data['returns'].shift(-1).values[:-1]  # Rendement futur comme cible
        X = X[:-1]  # Ajuster X pour correspondre à y
        
        return X, y
    
    def train(self, data, epochs=10, eval_interval=2):
        """
        Entraîne le système de trading sur les données historiques.
        
        Args:
            data (pd.DataFrame): Données de marché historiques
            epochs (int): Nombre d'époques d'entraînement
            eval_interval (int): Intervalle d'évaluation
            
        Returns:
            nn.Module: Meilleur réseau entraîné
        """
        self.logger.info(f"Début de l'entraînement sur {len(data)} points de données")
        
        # Prétraiter les données
        X, y = self.preprocess_data(data)
        
        # Convertir en tenseurs PyTorch
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Entraînement évolutionnaire
        for epoch in range(epochs):
            self.logger.info(f"Époque {epoch+1}/{epochs}")
            
            # Évaluer la population
            best_individual, best_fitness = self.population.evaluate_population(X, y)
            
            # Mettre à jour le meilleur réseau si nécessaire
            if best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                self.best_network = best_individual.develop_network()
                self.logger.info(f"Nouveau meilleur réseau trouvé: fitness={best_fitness:.4f}")
                
                # Sauvegarder le meilleur modèle
                model_path = os.path.join(self.model_dir, f"best_network_epoch_{epoch+1}.pt")
                torch.save(self.best_network.state_dict(), model_path)
            
            # Enregistrer les performances
            self.performance_history.append({
                'epoch': epoch + 1,
                'best_fitness': self.best_fitness,
                'avg_fitness': sum(self.population.fitness_scores) / len(self.population.fitness_scores),
                'diversity': self.population.get_diversity()
            })
            
            # Faire évoluer la population
            self.population.evolve()
            
            # Adapter les paramètres évolutionnaires
            if epoch % eval_interval == 0:
                self.population.adapt_parameters()
                
                # Stocker l'expérience pour le méta-apprentissage
                if epoch > 0:
                    # Extraire les caractéristiques du marché
                    market_features = self._extract_market_features(data)
                    
                    # Stocker l'expérience
                    self.meta_learner.store_experience(
                        state=market_features,
                        action=epoch % 3,  # Action simplifiée
                        reward=best_fitness,
                        next_state=market_features,  # Simplifié
                        hyperparams=np.array([self.population.mutation_prob, self.population.crossover_prob, 0.001, 0.2]),
                        performance=np.array([best_fitness])
                    )
                    
                    # Apprentissage méta
                    self.meta_learner.learn()
        
        # Visualiser l'évolution des performances
        self._plot_training_history()
        
        self.logger.info(f"Entraînement terminé. Meilleure fitness: {self.best_fitness:.4f}")
        
        return self.best_network
    
    def predict(self, data):
        """
        Fait des prédictions avec le meilleur réseau.
        
        Args:
            data (pd.DataFrame): Données de marché récentes
            
        Returns:
            np.array: Prédictions
        """
        if self.best_network is None:
            self.logger.error("Aucun réseau entraîné disponible pour la prédiction")
            return None
        
        # Prétraiter les données
        X, _ = self.preprocess_data(data.copy())
        
        # Convertir en tenseur PyTorch
        X_tensor = torch.FloatTensor(X)
        
        # Extraire les indicateurs pour la fusion
        rsi = torch.FloatTensor(data['rsi'].values[-len(X):]).unsqueeze(1)
        macd = torch.FloatTensor(data[['macd', 'macd_signal', 'macd_hist']].values[-len(X):])
        bollinger = torch.FloatTensor(data[['bb_width']].values[-len(X):])
        
        # Fusion des signaux
        fused_signal = self.signal_fusion(rsi, macd, bollinger)
        
        # Prédiction avec le meilleur réseau
        with torch.no_grad():
            predictions = self.best_network(X_tensor)
            
            # Combiner avec le signal fusionné
            combined_predictions = (predictions + fused_signal) / 2
        
        return combined_predictions.numpy()
    
    def generate_trading_signals(self, predictions, threshold=0.0):
        """
        Génère des signaux de trading à partir des prédictions.
        
        Args:
            predictions (np.array): Prédictions du modèle
            threshold (float): Seuil pour les signaux
            
        Returns:
            np.array: Signaux de trading (1: achat, -1: vente, 0: neutre)
        """
        signals = np.zeros(len(predictions))
        
        # Générer les signaux
        signals[predictions > threshold] = 1  # Signal d'achat
        signals[predictions < -threshold] = -1  # Signal de vente
        
        return signals
    
    def adapt_to_market(self, data):
        """
        Adapte le système aux conditions actuelles du marché.
        
        Args:
            data (pd.DataFrame): Données de marché récentes
            
        Returns:
            dict: Recommandations d'adaptation
        """
        # Utiliser le méta-apprenant pour adapter le système
        recommendations = self.meta_learner.adapt_to_market(data, None, None)
        
        # Appliquer les recommandations
        if 'mutation_rate' in recommendations:
            for individual in self.population.population:
                individual.mutation_rate = recommendations['mutation_rate']
        
        self.logger.info(f"Système adapté aux conditions de marché: {recommendations}")
        
        return recommendations
    
    def _extract_market_features(self, data):
        """
        Extrait les caractéristiques du marché pour le méta-apprentissage.
        
        Args:
            data (pd.DataFrame): Données de marché
            
        Returns:
            np.array: Caractéristiques du marché
        """
        # Utiliser les dernières données disponibles
        recent_data = data.tail(100)
        
        # Calculer les rendements
        returns = recent_data['returns'].dropna().values
        
        # Calculer la volatilité
        volatility = np.std(returns)
        
        # Calculer la tendance
        trend = np.mean(returns)
        
        # Utiliser les indicateurs techniques moyens
        rsi_mean = recent_data['rsi'].mean() / 100  # Normaliser
        macd_mean = recent_data['macd'].mean()
        bb_width_mean = recent_data['bb_width'].mean()
        
        # Créer un vecteur de caractéristiques
        features = np.array([
            trend,
            volatility,
            rsi_mean,
            macd_mean,
            bb_width_mean,
            np.percentile(returns, 10),  # Risque de perte
            np.percentile(returns, 90),  # Potentiel de gain
            np.mean(returns[-5:]),  # Tendance récente
            np.std(returns[-5:]),   # Volatilité récente
            recent_data['sma_ratio'].mean() - 1  # Écart moyen entre moyennes mobiles
        ])
        
        return features
    
    def _plot_training_history(self):
        """
        Visualise l'historique d'entraînement.
        """
        if not self.performance_history:
            return
        
        # Extraire les données
        epochs = [p['epoch'] for p in self.performance_history]
        best_fitness = [p['best_fitness'] for p in self.performance_history]
        avg_fitness = [p['avg_fitness'] for p in self.performance_history]
        diversity = [p['diversity'] for p in self.performance_history]
        
        # Créer la figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Tracer la fitness
        ax1.plot(epochs, best_fitness, 'b-', label='Meilleure fitness')
        ax1.plot(epochs, avg_fitness, 'r--', label='Fitness moyenne')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Évolution de la fitness pendant l\'entraînement')
        ax1.legend()
        ax1.grid(True)
        
        # Tracer la diversité
        ax2.plot(epochs, diversity, 'g-', label='Diversité génétique')
        ax2.set_xlabel('Époque')
        ax2.set_ylabel('Diversité')
        ax2.set_title('Évolution de la diversité génétique')
        ax2.legend()
        ax2.grid(True)
        
        # Sauvegarder la figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_history.png'))
        plt.close()
        
        self.logger.info(f"Historique d'entraînement visualisé et sauvegardé")