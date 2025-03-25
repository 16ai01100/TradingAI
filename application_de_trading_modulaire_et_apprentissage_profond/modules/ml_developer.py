import pandas as pd
import numpy as np
import logging
import os
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# Tensorflow et Keras pour l'apprentissage profond
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Conv1D, MaxPooling1D, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class MLDeveloper:
    """
    Module d'apprentissage profond pour prédire les mouvements du marché et améliorer les stratégies de trading.
    """
    
    def __init__(self, ml_framework='tensorflow', performance_metrics=None):
        """
        Initialise le module d'apprentissage profond.
        
        Args:
            ml_framework (str): Framework d'apprentissage machine à utiliser ('tensorflow' ou 'pytorch')
            performance_metrics (dict): Métriques de performance pour l'auto-apprentissage
        """
        self.ml_framework = ml_framework
        self.performance_metrics = performance_metrics or {}
        self.models = {}
        self.scalers = {}
        self.history = {}
        self.model_dir = os.path.join(os.getcwd(), 'models')
        self.log_dir = os.path.join(os.getcwd(), 'logs')
        
        # Création des répertoires s'ils n'existent pas
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Configuration du logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'ml_developer.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('MLDeveloper')
        self.logger.info(f"Module d'apprentissage profond initialisé avec {ml_framework}")
    
    def preprocess_data(self, data, crypto_pair, features=None, target='close', sequence_length=60, test_size=0.2):
        """
        Prétraite les données pour l'apprentissage profond.
        
        Args:
            data (pd.DataFrame): Données de marché
            crypto_pair (str): Paire de crypto-monnaies
            features (list): Liste des caractéristiques à utiliser
            target (str): Variable cible à prédire
            sequence_length (int): Longueur de la séquence pour les modèles LSTM
            test_size (float): Proportion des données pour le test
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, scaler)
        """
        self.logger.info(f"Prétraitement des données pour {crypto_pair}")
        
        # Sélection des caractéristiques
        if features is None:
            features = ['open', 'high', 'low', 'close', 'volume']
        
        # Vérification des données manquantes
        if data[features + [target]].isnull().any().any():
            self.logger.warning(f"Données manquantes détectées pour {crypto_pair}")
            data = data.dropna()
        
        # Normalisation des données
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[features + [target]])
        self.scalers[crypto_pair] = scaler
        
        # Création des séquences pour LSTM
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, :-1])
            y.append(scaled_data[i, -1])
        
        X, y = np.array(X), np.array(y)
        
        # Division en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        
        self.logger.info(f"Prétraitement terminé. Forme des données: X_train {X_train.shape}, y_train {y_train.shape}")
        
        return X_train, X_test, y_train, y_test, scaler
    
    def build_lstm_model(self, input_shape, units=50, dropout_rate=0.2, learning_rate=0.001):
        """
        Construit un modèle LSTM pour la prédiction de séries temporelles.
        
        Args:
            input_shape (tuple): Forme des données d'entrée
            units (int): Nombre d'unités LSTM
            dropout_rate (float): Taux de dropout
            learning_rate (float): Taux d'apprentissage
            
        Returns:
            tf.keras.Model: Modèle LSTM compilé
        """
        self.logger.info(f"Construction d'un modèle LSTM avec {units} unités")
        
        model = Sequential()
        model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units=units, return_sequences=False))
        model.add(Dropout(dropout_rate))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
        
        self.logger.info(f"Modèle LSTM compilé")
        
        return model
    
    def build_cnn_lstm_model(self, input_shape, cnn_filters=64, lstm_units=50, dropout_rate=0.2, learning_rate=0.001):
        """
        Construit un modèle hybride CNN-LSTM pour la prédiction de séries temporelles.
        
        Args:
            input_shape (tuple): Forme des données d'entrée
            cnn_filters (int): Nombre de filtres CNN
            lstm_units (int): Nombre d'unités LSTM
            dropout_rate (float): Taux de dropout
            learning_rate (float): Taux d'apprentissage
            
        Returns:
            tf.keras.Model: Modèle CNN-LSTM compilé
        """
        self.logger.info(f"Construction d'un modèle CNN-LSTM avec {cnn_filters} filtres et {lstm_units} unités LSTM")
        
        model = Sequential()
        model.add(Conv1D(filters=cnn_filters, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(units=lstm_units, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units=lstm_units, return_sequences=False))
        model.add(Dropout(dropout_rate))
        model.add(Dense(units=1))
        
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
        
        self.logger.info(f"Modèle CNN-LSTM compilé")
        
        return model
    
    def train_model(self, model, X_train, y_train, X_test, y_test, crypto_pair, model_type='lstm', 
                   batch_size=32, epochs=100, patience=10):
        """
        Entraîne un modèle d'apprentissage profond.
        
        Args:
            model (tf.keras.Model): Modèle à entraîner
            X_train (np.array): Données d'entraînement
            y_train (np.array): Cibles d'entraînement
            X_test (np.array): Données de test
            y_test (np.array): Cibles de test
            crypto_pair (str): Paire de crypto-monnaies
            model_type (str): Type de modèle ('lstm' ou 'cnn_lstm')
            batch_size (int): Taille du batch
            epochs (int): Nombre d'époques
            patience (int): Patience pour l'arrêt précoce
            
        Returns:
            tf.keras.Model: Modèle entraîné
        """
        self.logger.info(f"Entraînement du modèle {model_type} pour {crypto_pair}")
        
        # Callbacks pour l'entraînement
        model_path = os.path.join(self.model_dir, f"{crypto_pair}_{model_type}_model.h5")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)
        ]
        
        # Entraînement du modèle
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Sauvegarde du modèle et de l'historique
        self.models[f"{crypto_pair}_{model_type}"] = model
        self.history[f"{crypto_pair}_{model_type}"] = history.history
        
        # Sauvegarde du scaler
        joblib.dump(self.scalers[crypto_pair], os.path.join(self.model_dir, f"{crypto_pair}_scaler.pkl"))
        
        self.logger.info(f"Entraînement terminé pour {crypto_pair}_{model_type}")
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, scaler, crypto_pair, model_type='lstm', features_count=4):
        """
        Évalue les performances du modèle.
        
        Args:
            model (tf.keras.Model): Modèle à évaluer
            X_test (np.array): Données de test
            y_test (np.array): Cibles de test
            scaler (MinMaxScaler): Scaler utilisé pour la normalisation
            crypto_pair (str): Paire de crypto-monnaies
            model_type (str): Type de modèle ('lstm' ou 'cnn_lstm')
            features_count (int): Nombre de caractéristiques utilisées
            
        Returns:
            dict: Métriques d'évaluation
        """
        self.logger.info(f"Évaluation du modèle {model_type} pour {crypto_pair}")
        
        # Prédictions sur l'ensemble de test
        predictions = model.predict(X_test)
        
        # Dénormalisation des prédictions
        dummy_array = np.zeros((len(predictions), features_count + 1))
        dummy_array[:, -1] = predictions.flatten()
        predictions_denormalized = scaler.inverse_transform(dummy_array)[:, -1]
        
        dummy_array = np.zeros((len(y_test), features_count + 1))
        dummy_array[:, -1] = y_test
        y_test_denormalized = scaler.inverse_transform(dummy_array)[:, -1]
        
        # Calcul des métriques
        mse = mean_squared_error(y_test_denormalized, predictions_denormalized)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_denormalized, predictions_denormalized)
        r2 = r2_score(y_test_denormalized, predictions_denormalized)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        self.logger.info(f"Métriques pour {crypto_pair}_{model_type}: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        
        # Visualisation des prédictions
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_denormalized, label='Réel')
        plt.plot(predictions_denormalized, label='Prédictions')
        plt.title(f'Prédictions vs Réel pour {crypto_pair} avec {model_type}')
        plt.xlabel('Temps')
        plt.ylabel('Prix')
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, f"{crypto_pair}_{model_type}_predictions.png"))
        
        return metrics
    
    def predict(self, data, crypto_pair, model_type='lstm', sequence_length=60):
        """
        Fait des prédictions avec le modèle entraîné.
        
        Args:
            data (pd.DataFrame): Données récentes pour la prédiction
            crypto_pair (str): Paire de crypto-monnaies
            model_type (str): Type de modèle ('lstm' ou 'cnn_lstm')
            sequence_length (int): Longueur de la séquence pour les modèles LSTM
            
        Returns:
            float: Prédiction du prix
        """
        self.logger.info(f"Prédiction pour {crypto_pair} avec le modèle {model_type}")
        
        # Chargement du modèle et du scaler si nécessaire
        model_key = f"{crypto_pair}_{model_type}"
        if model_key not in self.models:
            model_path = os.path.join(self.model_dir, f"{model_key}_model.h5")
            if os.path.exists(model_path):
                self.models[model_key] = load_model(model_path)
                self.logger.info(f"Modèle chargé depuis {model_path}")
            else:
                self.logger.error(f"Modèle {model_key} non trouvé")
                return None
        
        if crypto_pair not in self.scalers:
            scaler_path = os.path.join(self.model_dir, f"{crypto_pair}_scaler.pkl")
            if os.path.exists(scaler_path):
                self.scalers[crypto_pair] = joblib.load(scaler_path)
                self.logger.info(f"Scaler chargé depuis {scaler_path}")
            else:
                self.logger.error(f"Scaler pour {crypto_pair} non trouvé")
                return None
        
        # Préparation des données pour la prédiction
        features = data.columns.tolist()
        if 'close' in features:
            features.remove('close')
        
        scaled_data = self.scalers[crypto_pair].transform(data)
        X_pred = np.array([scaled_data[-sequence_length:, :-1]])
        
        # Prédiction
        prediction = self.models[model_key].predict(X_pred)
        
        # Dénormalisation de la prédiction
        dummy_array = np.zeros((1, len(features) + 1))
        dummy_array[0, -1] = prediction[0, 0]
        prediction_denormalized = self.scalers[crypto_pair].inverse_transform(dummy_array)[0, -1]
        
        self.logger.info(f"Prédiction pour {crypto_pair}: {prediction_denormalized:.4f}")
        
        return prediction_denormalized
    
    def auto_learning(self, new_data, crypto_pair, actual_price, model_type='lstm', 
                      retrain_threshold=0.05, sequence_length=60, batch_size=32, epochs=10):
        """
        Implémente l'auto-apprentissage en retrainant le modèle avec de nouvelles données.
        
        Args:
            new_data (pd.DataFrame): Nouvelles données de marché
            crypto_pair (str): Paire de crypto-monnaies
            actual_price (float): Prix réel observé
            model_type (str): Type de modèle ('lstm' ou 'cnn_lstm')
            retrain_threshold (float): Seuil d'erreur pour le retraining
            sequence_length (int): Longueur de la séquence pour les modèles LSTM
            batch_size (int): Taille du batch pour le retraining
            epochs (int): Nombre d'époques pour le retraining
            
        Returns:
            bool: True si le modèle a été retrainé, False sinon
        """
        self.logger.info(f"Auto-apprentissage pour {crypto_pair} avec le modèle {model_type}")
        
        # Faire une prédiction avec les données actuelles
        predicted_price = self.predict(new_data, crypto_pair, model_type, sequence_length)
        if predicted_price is None:
            return False
        
        # Calculer l'erreur relative
        relative_error = abs(predicted_price - actual_price) / actual_price
        self.logger.info(f"Erreur relative pour {crypto_pair}: {relative_error:.4f}")
        
        # Si l'erreur est supérieure au seuil, retrainer le modèle
        if relative_error > retrain_threshold:
            self.logger.info(f"Retraining du modèle {model_type} pour {crypto_pair} (erreur: {relative_error:.4f})")
            
            # Prétraitement des nouvelles données
            features = new_data.columns.tolist()
            if 'close' in features:
                target = 'close'
                features.remove('close')
            else:
                target = features[-1]
                features = features[:-1]
            
            # Préparation des données pour le retraining
            X_train, X_test, y_train, y_test, _ = self.preprocess_data(
                new_data, crypto_pair, features, target, sequence_length, test_size=0.1
            )
            
            # Retraining du modèle
            model_key = f"{crypto_pair}_{model_type}"
            model = self.models[model_key]
            
            history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_test, y_test),
                verbose=1
            )
            
            # Mise à jour de l'historique
            if model_key in self.history:
                for key in history.history:
                    self.history[model_key][key].extend(history.history[key])
            else:
                self.history[model_key] = history.history
            
            # Sauvegarde du modèle mis à jour
            model_path = os.path.join(self.model_dir, f"{model_key}_model.h5")
            model.save(model_path)
            self.logger.info(f"Modèle mis à jour sauvegardé dans {model_path}")
            
            return True
        
        return False
    
    def optimize_hyperparameters(self, data, crypto_pair, features=None, target='close', 
                               sequence_length=60, test_size=0.2, model_type='lstm'):
        """
        Optimise les hyperparamètres du modèle.
        
        Args:
            data (pd.DataFrame): Données de marché
            crypto_pair (str): Paire de crypto-monnaies
            features (list): Liste des caractéristiques à utiliser
            target (str): Variable cible à prédire
            sequence_length (int): Longueur de la séquence pour les modèles LSTM
            test_size (float): Proportion des données pour le test
            model_type (str): Type de modèle ('lstm' ou 'cnn_lstm')
            
        Returns:
            dict: Meilleurs hyperparamètres
        """
        self.logger.info(f"Optimisation des hyperparamètres pour {crypto_pair} avec le modèle {model_type}")
        
        # Prétraitement des données
        X_train, X_test, y_train, y_test, scaler = self.preprocess_data(
            data, crypto_pair, features, target, sequence_length, test_size
        )
        
        # Hyperparamètres à tester
        if model_type == 'lstm':
            hyperparams = {
                'units': [32, 50, 64, 128],
                'dropout_rate': [0.1, 0.2, 0.3],
                'learning_rate': [0.001, 0.0005, 0.0001],
                'batch_size': [16, 32, 64]
            }
        else:  # cnn_lstm
            hyperparams = {
                'cnn_filters': [32, 64, 128],
                'lstm_units': [32, 50, 64],
                'dropout_rate': [0.1, 0.2, 0.3],
                'learning_rate': [0.001, 0.0005, 0.0001],
                'batch_size': [16, 32, 64]
            }
        
        # Recherche par grille simplifiée
        best_rmse = float('inf')
        best_params = {}
        
        # Limiter le nombre de combinaisons pour éviter une explosion combinatoire
        max_combinations = 5
        combinations = 0
        
        # Générer quelques combinaisons aléatoires d'hyperparamètres
        import random
        for _ in range(max_combinations):
            if model_type == 'lstm':
                params = {
                    'units': random.choice(hyperparams['units']),
                    'dropout_rate': random.choice(hyperparams['dropout_rate']),
                    'learning_rate': random.choice(hyperparams['learning_rate']),
                    'batch_size': random.choice(hyperparams['batch_size'])
                }
                
                # Construire et entraîner le modèle
                model = self.build_lstm_model(
                    input_shape=(X_train.shape[1], X_train.shape[2]),
                    units=params['units'],
                    dropout_rate=params['dropout_rate'],
                    learning_rate=params['learning_rate']
                )
            else:  # cnn_lstm
                params = {
                    'cnn_filters': random.choice(hyperparams['cnn_filters']),
                    'lstm_units': random.choice(hyperparams['lstm_units']),
                    'dropout_rate': random.choice(hyperparams['dropout_rate']),
                    'learning_rate': random.choice(hyperparams['learning_rate']),
                    'batch_size': random.choice(hyperparams['batch_size'])
                }
                
                # Construire et entraîner le modèle
                model = self.build_cnn_lstm_model(
                    input_shape=(X_train.shape[1], X_train.shape[2]),
                    cnn_filters=params['cnn_filters'],
                    lstm_units=params['lstm_units'],
                    dropout_rate=params['dropout_rate'],
                    learning_rate=params['learning_rate']
                )
            
            # Entraîner avec moins d'époques pour l'optimisation
            model = self.train_model(
                model, X_train, y_train, X_test, y_test,
                crypto_pair, model_type,
                batch_size=params['batch_size'],
                epochs=20,  # Moins d'époques pour l'optimisation
                patience=5
            )
            
            # Évaluer le modèle
            metrics = self.evaluate_model(
                model, X_test, y_test, scaler,
                crypto_pair, model_type,
                features_count=X_train.shape[2]
            )
            
            # Mettre à jour les meilleurs paramètres
            if metrics['rmse'] < best_rmse:
                best_rmse = metrics['rmse']
                best_params = params
            
            combinations += 1
        
        self.logger.info(f"Meilleurs hyperparamètres pour {crypto_pair}_{model_type}: {best_params}, RMSE: {best_rmse:.4f}")
        
        return best_params
    
    def generate_trading_signals(self, predictions, actual_prices, threshold=0.01):
        """
        Génère des signaux de trading basés sur les prédictions du modèle.
        
        Args:
            predictions (list): Liste des prédictions de prix
            actual_prices (list): Liste des prix réels
            threshold (float): Seuil de variation pour générer un signal
            
        Returns:
            list: Signaux de trading (1 pour achat, -1 pour vente, 0 pour conserver)
        """
        self.logger.info(f"Génération de signaux de trading avec seuil de {threshold}")
        
        signals = []
        for i in range(1, len(predictions)):
            # Calculer la variation prédite
            predicted_change = (predictions[i] - actual_prices[i-1]) / actual_prices[i-1]
            
            # Générer le signal
            if predicted_change > threshold:
                signals.append(1)  # Signal d'achat
            elif predicted_change < -threshold:
                signals.append(-1)  # Signal de vente
            else:
                signals.append(0)  # Conserver
        
        self.logger.info(f"Signaux générés: {len(signals)} (Achats: {signals.count(1)}, Ventes: {signals.count(-1)}, Conserver: {signals.count(0)})")
        
        return signals
    
    def save_model_performance(self, crypto_pair, model_type, metrics, predictions, actual_prices):
        """
        Sauvegarde les performances du modèle pour analyse ultérieure.
        
        Args:
            crypto_pair (str): Paire de crypto-monnaies
            model_type (str): Type de modèle ('lstm' ou 'cnn_lstm')
            metrics (dict): Métriques d'évaluation
            predictions (list): Liste des prédictions
            actual_prices (list): Liste des prix réels
        """
        self.logger.info(f"Sauvegarde des performances du modèle {model_type} pour {crypto_pair}")
        
        # Créer un DataFrame avec les prédictions et les prix réels
        performance_df = pd.DataFrame({
            'actual': actual_prices,
            'predicted': predictions,
            'error': np.array(predictions) - np.array(actual_prices),
            'error_pct': (np.array(predictions) - np.array(actual_prices)) / np.array(actual_prices) * 100
        })
        
        # Ajouter les métriques globales
        for key, value in metrics.items():
            performance_df[key] = value
        
        # Sauvegarder dans un fichier CSV
        performance_path = os.path.join(self.log_dir, f"{crypto_pair}_{model_type}_performance.csv")
        performance_df.to_csv(performance_path, index=False)
        
        self.logger.info(f"Performances sauvegardées dans {performance_path}")
        
        # Mettre à jour les métriques de performance pour l'auto-apprentissage
        self.performance_metrics[f"{crypto_pair}_{model_type}"] = metrics