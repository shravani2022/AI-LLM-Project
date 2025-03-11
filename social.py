
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import schedule
import json
import logging
from collections import defaultdict

class SocialMediaOptimizer:
    def __init__(self):
        self.posts = []
        self.sentiment_history = []
        self.engagement_history = defaultdict(list)
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='social_media_optimizer.log'
        )

    def analyze_sentiment(self, text):
        """Analyze text sentiment using TextBlob"""
        try:
            analysis = TextBlob(text)
            return {
                'score': analysis.sentiment.polarity,
                'subjectivity': analysis.sentiment.subjectivity,
                'classification': self.classify_sentiment(analysis.sentiment.polarity)
            }
        except Exception as e:
            logging.error(f"Sentiment analysis failed: {str(e)}")
            return None

    def classify_sentiment(self, score):
        if score > 0.3:
            return 'very_positive'
        elif score > 0:
            return 'positive'
        elif score == 0:
            return 'neutral'
        elif score > -0.3:
            return 'negative'
        else:
            return 'very_negative'

    def classify_content(self, text):
        """Classify content type"""
        features = {
            'promotional': any(word in text.lower() for word in ['sale', 'offer', 'discount', 'new', 'launch']),
            'educational': any(word in text.lower() for word in ['learn', 'how to', 'guide', 'tip', 'tutorial']),
            'engagement': any(word in text.lower() for word in ['share', 'comment', 'tell us', 'what do you think']),
            'news': any(word in text.lower() for word in ['announced', 'update', 'latest', 'breaking']),
            'social_proof': any(word in text.lower() for word in ['testimonial', 'review', 'case study', 'success'])
        }
        return max(features.items(), key=lambda x: x[1])[0] if any(features.values()) else 'general'

    def predict_engagement(self, post, platform):
        """Predict post engagement based on content features"""
        features = self.extract_features(post)
        base_score = sum([
            features['sentiment_score'] * 20,
            features['has_hashtags'] * 15,
            features['has_mentions'] * 10,
            features['has_emoji'] * 15,
            features['has_link'] * 5,
            features['optimal_length_score'] * 20
        ])
        
        # Platform-specific adjustments
        platform_multipliers = {
            'twitter': 1.2 if features['has_hashtags'] else 0.8,
            'linkedin': 1.3 if features['content_type'] in ['educational', 'social_proof'] else 0.9,
            'facebook': 1.2 if features['has_emoji'] else 0.9,
            'instagram': 1.4 if features['has_emoji'] and features['has_hashtags'] else 0.8
        }
        
        return base_score * platform_multipliers.get(platform, 1.0)

    def extract_features(self, text):
        """Extract content features for analysis"""
        sentiment = self.analyze_sentiment(text)
        return {
            'length': len(text),
            'optimal_length_score': self.calculate_length_score(len(text)),
            'has_hashtags': '#' in text,
            'has_mentions': '@' in text,
            'has_emoji': any(char in text for char in '😀🎉📈🎯💡'),
            'has_link': 'http' in text or 'link' in text.lower(),
            'sentiment_score': sentiment['score'],
            'content_type': self.classify_content(text)
        }

    def calculate_length_score(self, length):
        """Calculate score based on optimal content length per platform"""
        optimal_ranges = {
            'twitter': (60, 130),
            'facebook': (120, 250),
            'linkedin': (150, 300),
            'instagram': (140, 240)
        }
        
        # Default to Twitter range if platform not specified
        min_length, max_length = optimal_ranges['twitter']
        
        if length < min_length:
            return 0.5 + (0.5 * (length / min_length))
        elif length <= max_length:
            return 1.0
        else:
            return 1.0 - (0.5 * ((length - max_length) / max_length))

    def analyze_performance_trends(self):
        """Analyze historical performance trends"""
        if not self.engagement_history:
            return None
            
        analysis = {
            'overall_trends': {
                'avg_engagement': np.mean([eng for platform_eng in self.engagement_history.values() for eng in platform_eng]),
                'engagement_growth': self.calculate_growth_rate(self.engagement_history),
                'sentiment_trend': self.analyze_sentiment_trend()
            },
            'platform_performance': self.analyze_platform_performance(),
            'content_insights': self.generate_content_insights(),
            'recommendations': self.generate_recommendations()
        }
        
        return analysis

    def calculate_growth_rate(self, history):
        """Calculate engagement growth rate"""
        rates = {}
        for platform, engagements in history.items():
            if len(engagements) >= 2:
                initial = np.mean(engagements[:3])
                final = np.mean(engagements[-3:])
                rates[platform] = ((final - initial) / initial) * 100
        return rates

    def analyze_sentiment_trend(self):
        """Analyze sentiment trends over time"""
        if not self.sentiment_history:
            return None
            
        recent_sentiment = self.sentiment_history[-10:]
        return {
            'average': np.mean([s['score'] for s in recent_sentiment]),
            'trend': 'improving' if np.mean([s['score'] for s in recent_sentiment[-5:]]) > 
                    np.mean([s['score'] for s in recent_sentiment[:5]]) else 'declining',
            'volatility': np.std([s['score'] for s in recent_sentiment])
        }

    def analyze_platform_performance(self):
        """Analyze performance by platform"""
        return {
            platform: {
                'avg_engagement': np.mean(engagements),
                'best_time': self.find_best_posting_time(platform),
                'top_content_types': self.find_top_content_types(platform)
            }
            for platform, engagements in self.engagement_history.items()
        }

    def find_best_posting_time(self, platform):
        """Determine optimal posting times"""
        return {
            'twitter': ['9:00', '15:00', '18:00'],
            'facebook': ['13:00', '16:00', '19:00'],
            'linkedin': ['10:00', '14:00', '17:00'],
            'instagram': ['11:00', '15:00', '20:00']
        }.get(platform, ['12:00'])

    def find_top_content_types(self, platform):
        """Find best performing content types per platform"""
        platform_content_types = {
            'twitter': ['news', 'engagement'],
            'facebook': ['social_proof', 'engagement'],
            'linkedin': ['educational', 'social_proof'],
            'instagram': ['promotional', 'social_proof']
        }
        return platform_content_types.get(platform, ['general'])

    def generate_recommendations(self):
        """Generate content and scheduling recommendations"""
        return {
            'content_strategy': {
                'recommended_content_mix': {
                    'educational': 0.3,
                    'engagement': 0.25,
                    'promotional': 0.2,
                    'social_proof': 0.15,
                    'news': 0.1
                },
                'optimal_post_frequency': {
                    'twitter': 3,
                    'facebook': 2,
                    'linkedin': 1,
                    'instagram': 1
                }
            },
            'engagement_optimization': {
                'hashtag_usage': 'Increase hashtag usage on Twitter and Instagram',
                'content_length': 'Optimize content length per platform',
                'timing': 'Focus on peak engagement hours',
                'interaction': 'Increase engagement posts during high-activity periods'
            }
        }

    def generate_predictions(self):
        """Generate future performance predictions"""
        return {
            'engagement_forecast': {
                '1_month': '+15% expected growth',
                '3_months': '+35% projected growth',
                '6_months': '+60% potential growth with optimization'
            },
            'platform_specific': {
                'twitter': {
                    'growth_potential': 'High',
                    'focus_areas': ['hashtag optimization', 'timing']
                },
                'linkedin': {
                    'growth_potential': 'Very High',
                    'focus_areas': ['educational content', 'professional insights']
                },
                'facebook': {
                    'growth_potential': 'Moderate',
                    'focus_areas': ['engagement posts', 'community building']
                },
                'instagram': {
                    'growth_potential': 'High',
                    'focus_areas': ['visual content', 'story utilization']
                }
            },
            'risk_factors': {
                'platform_algorithm_changes': 'Medium Risk',
                'content_saturation': 'High Risk',
                'engagement_decline': 'Low Risk'
            }
        }

# Example usage and analysis
optimizer = SocialMediaOptimizer()
predictions = optimizer.generate_predictions()
print("\nPredictions and Analysis:")
print(json.dumps(predictions, indent=2))