import numpy as np
import pandas as pd
import tensorflow as tf
import os
import joblib
import logging
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import TimeSeriesSplit, train_test_split
import ta
from tensorflow import keras
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler
from datetime import datetime

class ScalpingAI:
    @staticmethod
    def weighted_categorical_crossentropy(weights=[1.0, 1.0, 1.0]):
        """Crée une fonction de perte avec pondération des classes"""
        weights = K.variable(weights)
        
        def loss(y_true, y_pred):
            y_true = K.cast(y_true, y_pred.dtype)
            return -K.mean(
                K.sum(y_true * weights * K.log(K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())), axis=-1)
            )
        
        return loss
    
    @staticmethod
    def precision_buy_sell(y_true, y_pred):
        """Calcule la précision pour les signaux Buy/Sell uniquement"""
        # Conversion des prédictions en classes
        y_pred_classes = K.argmax(y_pred, axis=-1)
        y_true_classes = K.argmax(y_true, axis=-1)
        
        # Masque pour Buy (2) et Sell (0)
        trade_mask = K.not_equal(y_true_classes, 1)  # Exclure Hold (1)
        
        # Calcul de la précision sur les trades
        correct_trades = K.cast(K.equal(y_pred_classes, y_true_classes), 'float32')
        trade_precision = K.sum(correct_trades * K.cast(trade_mask, 'float32')) / (K.sum(K.cast(trade_mask, 'float32')) + K.epsilon())
        
        return trade_precision
    
    @staticmethod
    def recall_buy_sell(y_true, y_pred):
        """Calcule le recall pour les signaux Buy/Sell uniquement"""
        # Conversion des prédictions en classes
        y_pred_classes = K.argmax(y_pred, axis=-1)
        y_true_classes = K.argmax(y_true, axis=-1)
        
        # Masques pour Buy (2) et Sell (0)
        true_trades = K.not_equal(y_true_classes, 1)  # Vrais trades (exclure Hold)
        pred_trades = K.not_equal(y_pred_classes, 1)  # Trades prédits
        
        # Calcul du recall
        correct_trades = K.cast(K.equal(y_pred_classes, y_true_classes), 'float32')
        trade_recall = K.sum(correct_trades * K.cast(true_trades, 'float32')) / (K.sum(K.cast(true_trades, 'float32')) + K.epsilon())
        
        return trade_recall

    def __init__(self, model_path='models/model.h5', scaler_path='models/scaler.joblib'):
        """Initialise le modèle"""
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.logger = logging.getLogger('ScalpingAI')
        
        try:
            if os.path.exists(model_path):
                # On ne charge pas le modèle tout de suite, on attend d'avoir la bonne shape
                self.logger.info("Modèle existant trouvé")
            else:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                self.logger.info("Aucun modèle existant trouvé")
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation : {str(e)}")
            raise

    def setup_logging(self):
        """Configure le système de logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/ai_model.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ScalpingAI')
    
    def preprocess_data(self, df):
        """Prétraitement avec seuils plus sensibles"""
        try:
            # Features techniques
            df['price_change'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            df['momentum'] = df['close'].diff(3)
            df['volatility'] = df['close'].rolling(window=10).std()
            
            # Seuils très sensibles
            price_threshold = 0.001  # 0.1%
            volume_threshold = 1.2   # 20%
            
            # Conditions d'achat plus souples
            buy_conditions = (
                (df['price_change'] > price_threshold) & 
                (df['volume_change'] > volume_threshold)
            ) | (
                (df['price_change'] > price_threshold * 1.5) &
                (df['momentum'] > 0)
            )
            
            # Conditions de vente plus souples
            sell_conditions = (
                (df['price_change'] < -price_threshold) & 
                (df['volume_change'] > volume_threshold)
            ) | (
                (df['price_change'] < -price_threshold * 1.5) &
                (df['momentum'] < 0)
            )
            
            # Génération des signaux
            df['target'] = 1  # Hold par défaut
            df.loc[buy_conditions, 'target'] = 2  # Buy
            df.loc[sell_conditions, 'target'] = 0  # Sell
            
            # Liste des features à normaliser
            feature_columns = [
                'price_change', 'volume_change', 'momentum', 'volatility'
            ]
            
            # Normalisation
            X = df[feature_columns].copy()
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            if self.scaler is None:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
                joblib.dump(self.scaler, self.scaler_path)
                self.logger.info("Nouveau scaler créé et sauvegardé")
            else:
                try:
                    X_scaled = self.scaler.transform(X)
                except Exception as e:
                    self.logger.warning(f"Recréation du scaler due à : {str(e)}")
                    self.scaler = StandardScaler()
                    X_scaled = self.scaler.fit_transform(X)
                    joblib.dump(self.scaler, self.scaler_path)
            
            # Mise à jour des features normalisées
            for i, col in enumerate(feature_columns):
                df[col] = X_scaled[:, i]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erreur lors du prétraitement : {str(e)}")
            raise

    def create_model(self, input_shape):
        """Crée un modèle LSTM optimisé pour le scalping"""
        try:
            self.logger.info(f"Création du modèle avec input_shape={input_shape}")
            
            # Configuration optimisée
            lstm_units = [512, 256, 128]
            dense_units = [128, 64]
            dropout_rates = [0.4, 0.4, 0.3]
            l2_reg = 0.001
            
            inputs = tf.keras.Input(shape=input_shape)
            
            # LSTM plus profond
            x = tf.keras.layers.LSTM(
                lstm_units[0],
                return_sequences=True,
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
            )(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rates[0])(x)
            
            x = tf.keras.layers.LSTM(
                lstm_units[1],
                return_sequences=True,
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rates[1])(x)
            
            x = tf.keras.layers.LSTM(
                lstm_units[2],
                return_sequences=False,
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rates[2])(x)
            
            # Couches denses plus larges
            for units in dense_units:
                x = tf.keras.layers.Dense(
                    units,
                    activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
                )(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Dropout(0.3)(x)
            
            outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
            
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            # Optimiseur avec learning rate plus agressif
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
            
            model.compile(
                optimizer=optimizer,
                loss=self.weighted_categorical_crossentropy([5.0, 0.2, 5.0]),
                metrics=['accuracy', self.precision_buy_sell, self.recall_buy_sell]
            )
            
            model.summary(print_fn=self.logger.info)
            return model
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la création du modèle : {str(e)}")
            raise

    def load_or_create_model(self, input_shape):
        """Charge ou crée le modèle avec la bonne shape"""
        try:
            if os.path.exists(self.model_path):
                try:
                    # Charger le modèle existant
                    self.model = tf.keras.models.load_model(
                        self.model_path,
                        custom_objects={
                            'precision_buy_sell': self.precision_buy_sell,
                            'recall_buy_sell': self.recall_buy_sell,
                            'weighted_categorical_crossentropy': self.weighted_categorical_crossentropy([5.0, 0.2, 5.0])
                        }
                    )
                    
                    # Vérifier si la shape correspond
                    expected_shape = self.model.input_shape[1:]
                    if expected_shape == input_shape:
                        self.logger.info("Modèle chargé avec succès")
                        return self.model
                    else:
                        self.logger.warning(f"Shape du modèle existant ({expected_shape}) différente de la shape attendue ({input_shape})")
                        self.logger.info("Création d'un nouveau modèle...")
                except Exception as e:
                    self.logger.warning(f"Erreur lors du chargement du modèle : {str(e)}")
                    self.logger.info("Création d'un nouveau modèle...")
            
            # Créer un nouveau modèle
            self.model = self.create_model(input_shape)
            self.logger.info("Nouveau modèle créé avec succès")
            return self.model
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement/création du modèle : {str(e)}")
            raise

    def train(self, df, epochs=100, batch_size=32):
        """Entraînement optimisé pour le scalping"""
        try:
            # Prétraitement et préparation des données
            df = self.preprocess_data(df)
            X, y = self.prepare_sequences(df)
            
            # Initialisation du modèle avec la bonne shape
            if self.model is None:
                self.model = self.load_or_create_model(X.shape[1:])
            
            # Poids des classes très déséquilibrés
            class_weights = {
                0: 5.0,  # Sell
                1: 0.2,  # Hold
                2: 5.0   # Buy
            }
            
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_precision_buy_sell',
                    patience=20,
                    restore_best_weights=True,
                    mode='max'
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_precision_buy_sell',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    mode='max'
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    'models/best_model.h5',
                    monitor='val_precision_buy_sell',
                    save_best_only=True,
                    mode='max'
                )
            ]
            
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=callbacks,
                class_weight=class_weights,
                shuffle=True
            )
            
            # Sauvegarde du modèle final
            self.model.save(self.model_path)
            self.logger.info("Modèle sauvegardé avec succès")
            
            return history
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'entraînement : {str(e)}")
            raise

    def prepare_sequences(self, df, sequence_length=60):
        """Prépare les séquences pour l'entraînement"""
        try:
            # Features à utiliser pour l'entraînement
            feature_columns = [
                'price_change', 'volume_change', 'momentum', 'volatility'
            ]
            
            # Création des séquences
            sequences = []
            targets = []
            
            # Pour chaque timeframe
            for timeframe in df['timeframe'].unique():
                # Filtrer les données pour ce timeframe
                timeframe_data = df[df['timeframe'] == timeframe]
                
                # Création des séquences pour ce timeframe
                for i in range(len(timeframe_data) - sequence_length):
                    seq = timeframe_data[feature_columns].iloc[i:i + sequence_length].values
                    target = timeframe_data['target'].iloc[i + sequence_length - 1]
                    
                    # Convertir target en one-hot encoding
                    target_one_hot = np.zeros(3)
                    target_one_hot[int(target)] = 1
                    
                    sequences.append(seq)
                    targets.append(target_one_hot)
            
            # Conversion en arrays numpy
            X = np.array(sequences)
            y = np.array(targets)
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la préparation des séquences : {str(e)}")
            raise

    def predict(self, df):
        """Génère des prédictions de trading"""
        try:
            # Prétraiter les données
            df = self.preprocess_data(df)
            
            # Préparer les séquences
            sequences, _ = self.prepare_sequences(df)
            
            if len(sequences) == 0:
                raise ValueError("Pas assez de données pour faire une prédiction")
            
            # Faire la prédiction
            predictions = self.model.predict(sequences, verbose=0)
            
            # Appliquer le seuil de confiance adaptatif
            probs_sell = predictions[..., 0]
            probs_buy = predictions[..., 2]
            max_probs = np.maximum(probs_sell, probs_buy)
            
            # Convertir en classes
            pred_classes = np.argmax(predictions, axis=1)
            
            # Appliquer le seuil adaptatif
            pred_classes[max_probs < 0.4] = 1  # Hold si confiance insuffisante
            
            # Calculer la confiance et l'action
            confidence = np.max(predictions[-1])
            action_idx = np.argmax(predictions[-1])
            
            # Vérifier les conditions supplémentaires
            last_data = df.iloc[-1]
            action = self.validate_prediction(action_idx, confidence, last_data)
            
            return {
                'action': action,
                'confidence': float(confidence),
                'raw_predictions': predictions[-1].tolist(),
                'timestamp': last_data['timestamp'],
                'close_price': float(last_data['close']),
                'indicators': {
                    'rsi': float(last_data['rsi']),
                    'macd': float(last_data['macd']),
                    'macd_signal': float(last_data['macd_signal']),
                    'volume_ratio': float(last_data['volume_ratio']),
                    'volatility': float(last_data['volatility'])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la prédiction : {str(e)}")
            raise
            
    def validate_prediction(self, action_idx, confidence, data):
        """Valide et ajuste la prédiction en fonction des conditions du marché"""
        try:
            # Mapping des actions
            actions = ['sell', 'hold', 'buy']
            action = actions[action_idx]
            
            # Seuils de confiance minimale
            MIN_CONFIDENCE = 0.6
            
            # Si la confiance est trop faible, hold
            if confidence < MIN_CONFIDENCE:
                return 'hold'
            
            # Vérifier les conditions de marché
            rsi = data['rsi']
            macd = data['macd']
            macd_signal = data['macd_signal']
            volume_ratio = data['volume_ratio']
            volatility = data['volatility']
            
            if action == 'buy':
                # Éviter d'acheter en surachat ou volume faible
                if rsi > 70 or volume_ratio < 0.8:
                    return 'hold'
                # Confirmer avec MACD
                if macd <= macd_signal:
                    return 'hold'
                    
            elif action == 'sell':
                # Éviter de vendre en survente ou volume faible
                if rsi < 30 or volume_ratio < 0.8:
                    return 'hold'
                # Confirmer avec MACD
                if macd >= macd_signal:
                    return 'hold'
            
            # Vérifier la volatilité
            if volatility > 0.02:  
                self.logger.warning(f"Volatilité élevée ({volatility:.2%}), prudence recommandée")
            
            return action
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la validation : {str(e)}")
            return 'hold'  

    def update_confidence_threshold(self):
        """Ajuste le seuil de confiance en fonction des performances"""
        if len(self.performance_history['precision']) >= 5:
            recent_precision = np.mean(self.performance_history['precision'][-5:])
            recent_recall = np.mean(self.performance_history['recall'][-5:])
            
            # Si précision trop basse, augmenter le seuil
            if recent_precision < 0.5:
                self.confidence_threshold = min(self.confidence_threshold + 0.02, 0.7)
            # Si recall trop bas et précision ok, diminuer le seuil
            elif recent_recall < 0.3 and recent_precision > 0.6:
                self.confidence_threshold = max(self.confidence_threshold - 0.02, 0.3)
            
            self.logger.info(f"Seuil de confiance ajusté à : {self.confidence_threshold:.2f}")

    def add_to_training_buffer(self, X, y, actual_profit=None):
        """Ajoute des données au buffer d'entraînement"""
        self.training_buffer['X'].extend(X)
        self.training_buffer['y'].extend(y)
        
        # Si le buffer est plein, retirer les plus anciennes données
        if len(self.training_buffer['X']) > self.training_buffer['max_size']:
            self.training_buffer['X'] = self.training_buffer['X'][-self.training_buffer['max_size']:]
            self.training_buffer['y'] = self.training_buffer['y'][-self.training_buffer['max_size']:]

    def incremental_train(self):
        """Entraînement incrémental sur le buffer"""
        if len(self.training_buffer['X']) < 100:  
            return
        
        X = np.array(self.training_buffer['X'])
        y = np.array(self.training_buffer['y'])
        
        # Entraînement sur une époque
        self.model.fit(
            X, y,
            epochs=1,
            batch_size=32,
            verbose=0
        )
        
        self.logger.info(f"Entraînement incrémental effectué sur {len(X)} échantillons")

    def evaluate_trade(self, prediction, actual_outcome, profit=None):
        """Évalue la performance d'une trade et met à jour les métriques"""
        if prediction in [0, 2]:  
            self.performance_history['trades'].append({
                'prediction': prediction,
                'outcome': actual_outcome,
                'profit': profit,
                'timestamp': datetime.now()
            })
            
            # Calculer les métriques récentes
            recent_trades = self.performance_history['trades'][-50:]  
            correct = sum(1 for t in recent_trades if t['prediction'] == t['outcome'])
            precision = correct / len(recent_trades) if recent_trades else 0
            
            self.performance_history['precision'].append(precision)
            
            # Mettre à jour le seuil de confiance
            self.update_confidence_threshold()
            
            # Entraînement incrémental si assez de données
            if len(self.training_buffer['X']) >= 100:
                self.incremental_train()
                
    def save_prediction_metrics(self, prediction, actual_return=None):
        """Sauvegarde les métriques de prédiction pour analyse"""
        try:
            metrics = {
                'timestamp': prediction['timestamp'],
                'action': prediction['action'],
                'confidence': prediction['confidence'],
                'close_price': prediction['close_price'],
                'rsi': prediction['indicators']['rsi'],
                'macd': prediction['indicators']['macd'],
                'volume_ratio': prediction['indicators']['volume_ratio'],
                'volatility': prediction['indicators']['volatility']
            }
            
            if actual_return is not None:
                metrics['actual_return'] = actual_return
            
            # Sauvegarder dans un fichier CSV
            metrics_file = os.path.join(os.path.dirname(self.model_path), 'prediction_metrics.csv')
            metrics_df = pd.DataFrame([metrics])
            
            if os.path.exists(metrics_file):
                metrics_df.to_csv(metrics_file, mode='a', header=False, index=False)
            else:
                metrics_df.to_csv(metrics_file, index=False)
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde des métriques : {str(e)}")
            
    def evaluate_model(self, data):
        """
        Évalue le modèle sur un jeu de données
        """
        try:
            # Prétraitement des données
            processed_data = self.preprocess_data(data)
            
            # Création des séquences
            X, y = self.prepare_sequences(processed_data)
            
            # Prédictions
            y_pred_proba = self.model.predict(X)
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_true = y
            
            # Obtenir les classes uniques présentes
            unique_classes = np.unique(np.concatenate([y_true, y_pred]))
            class_labels = ['Sell', 'Hold', 'Buy']
            present_labels = [class_labels[i] for i in unique_classes]
            
            # Calculer les métriques
            report = classification_report(y_true, y_pred, 
                                        target_names=present_labels,
                                        labels=unique_classes)
            
            self.logger.info(f"Rapport d'évaluation :\n{report}")
            
            # Calculer la matrice de confusion
            cm = confusion_matrix(y_true, y_pred)
            self.logger.info(f"Matrice de confusion :\n{cm}")
            
            return {
                'classification_report': report,
                'confusion_matrix': cm,
                'predictions': y_pred,
                'true_values': y_true
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'évaluation : {str(e)}")
            raise

if __name__ == "__main__":
    model = ScalpingAI()
    # Entraînement du modèle avec les données réelles et apprentissage continu
    df = pd.DataFrame({'timestamp': [1, 2, 3], 'open': [4, 5, 6], 'high': [7, 8, 9], 'low': [10, 11, 12], 'close': [13, 14, 15], 'volume': [16, 17, 18], 'timeframe': [1, 1, 1]})
    df = model.preprocess_data(df)
    X, y = model.prepare_sequences(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.train(df, epochs=100, batch_size=32)

    # Génération d'un signal de trading
    signal = model.predict(pd.DataFrame({'timestamp': [1, 2, 3], 'open': [4, 5, 6], 'high': [7, 8, 9], 'low': [10, 11, 12], 'close': [13, 14, 15], 'volume': [16, 17, 18], 'timeframe': [1, 1, 1]}))
    print(f"Signal de trading: {signal['action']}, Confiance: {signal['confidence']}")
