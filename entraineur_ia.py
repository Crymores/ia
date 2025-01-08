import os
import shutil
from datetime import datetime
import pandas as pd
import numpy as np
from ai_model import ScalpingAI
import ccxt
from dotenv import load_dotenv
import logging
import json

def setup_logging():
    """Configure le système de logging"""
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('Training')

def get_training_data(exchange, symbol, timeframes=['1m', '5m', '15m', '1h'], days=5):
    """Récupère les données d'entraînement de plusieurs timeframes"""
    logger = logging.getLogger('Training')
    all_data = []
    
    try:
        for timeframe in timeframes:
            logger.info(f"Récupération des données {timeframe} pour {symbol}...")
            
            # Calculer le nombre de bougies nécessaires
            if timeframe.endswith('m'):
                limit = min(1000, int(days * 24 * 60 / int(timeframe[:-1])))
            elif timeframe.endswith('h'):
                limit = min(1000, int(days * 24 / int(timeframe[:-1])))
            
            # Récupérer les données
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv or len(ohlcv) < 10:  # Vérifier qu'on a assez de données
                logger.warning(f"Pas assez de données pour {timeframe}, ignoré")
                continue
            
            # Convertir en DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timeframe'] = timeframe
            
            # Vérifier la qualité des données
            if df['volume'].sum() == 0:
                logger.warning(f"Aucun volume pour {timeframe}, ignoré")
                continue
                
            if df['close'].std() == 0:
                logger.warning(f"Pas de variation de prix pour {timeframe}, ignoré")
                continue
            
            all_data.append(df)
            logger.info(f"Récupéré {len(df)} bougies valides pour {timeframe}")
        
        if not all_data:
            raise ValueError("Aucune donnée valide récupérée pour tous les timeframes")
        
        # Fusionner toutes les données
        final_df = pd.concat(all_data, ignore_index=True)
        final_df['timestamp'] = pd.to_datetime(final_df['timestamp'], unit='ms')
        final_df = final_df.sort_values('timestamp')
        
        # Vérifier la qualité finale des données
        logger.info(f"Dataset final : {len(final_df)} lignes")
        logger.info(f"Timeframes présents : {final_df['timeframe'].unique()}")
        logger.info(f"Période : de {final_df['timestamp'].min()} à {final_df['timestamp'].max()}")
        
        return final_df
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données : {str(e)}")
        raise

def train_and_replace_model():
    """Entraîne un nouveau modèle et remplace l'ancien s'il est meilleur"""
    logger = setup_logging()
    
    try:
        logger.info("=== Démarrage de l'entraînement du modèle ===")
        
        # Créer les dossiers nécessaires
        os.makedirs('models', exist_ok=True)
        os.makedirs('models/backup', exist_ok=True)
        
        # Sauvegarder l'ancien modèle s'il existe
        current_model_path = 'models/best_model.h5'
        if os.path.exists(current_model_path):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f'models/backup/model_{timestamp}.h5'
            shutil.copy2(current_model_path, backup_path)
            logger.info(f"Ancien modèle sauvegardé : {backup_path}")
        
        # Initialiser l'exchange
        load_dotenv()
        exchange = ccxt.kucoin({
            'apiKey': os.getenv('KUCOIN_API_KEY'),
            'secret': os.getenv('KUCOIN_API_SECRET'),
            'password': os.getenv('KUCOIN_API_PASSPHRASE'),
            'enableRateLimit': True
        })
        
        # Récupérer les données d'entraînement
        symbol = 'SDM/USDT'
        logger.info(f"Récupération des données pour {symbol}...")
        df = get_training_data(exchange, symbol)
        
        # Créer et entraîner le modèle
        model = ScalpingAI()
        logger.info("Démarrage de l'entraînement...")
        
        history = model.train(df, epochs=50, batch_size=32)
        
        # Sauvegarder les métriques d'entraînement
        metrics = {
            'training_date': datetime.now().isoformat(),
            'symbol': symbol,
            'data_points': len(df),
            'final_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'final_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1])
        }
        
        with open('models/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info("Entraînement terminé avec succès!")
        logger.info(f"Métriques finales:")
        logger.info(f"- Loss: {metrics['final_loss']:.4f}")
        logger.info(f"- Validation Loss: {metrics['final_val_loss']:.4f}")
        logger.info(f"- Accuracy: {metrics['final_accuracy']:.4f}")
        logger.info(f"- Validation Accuracy: {metrics['final_val_accuracy']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement : {str(e)}")
        
        # Restaurer l'ancien modèle si l'entraînement a échoué
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, current_model_path)
            logger.info("Ancien modèle restauré")
        
        return False

if __name__ == '__main__':
    train_and_replace_model()
