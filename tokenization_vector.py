from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re

class TextVectorizer:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        except:
            pass
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer()
        self.pca = PCA(n_components=3)
        
    def preprocess_text(self, text):
        """
        Preprocess text by performing tokenization, removing stopwords,
        lemmatization, and cleaning
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words 
                 and token not in string.punctuation
                 and len(token) > 2]
        
        return tokens
    
    def vectorize_texts(self, texts):
        """
        Convert a list of texts into 3D vectors
        """
        # Preprocess all texts
        processed_texts = [' '.join(self.preprocess_text(text)) for text in texts]
        
        # Create TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
        
        # Reduce dimensionality to 3D using PCA
        vectors_3d = self.pca.fit_transform(tfidf_matrix.toarray())
        
        return vectors_3d
    
    def get_feature_importance(self):
        """
        Get the most important features (words) for each dimension
        """
        feature_names = self.vectorizer.get_feature_names_out()
        components = self.pca.components_
        
        important_features = []
        for i, component in enumerate(components):
            # Get indices of top 5 words for this component
            top_indices = component.argsort()[-5:][::-1]
            top_words = [(feature_names[idx], component[idx]) for idx in top_indices]
            important_features.append(top_words)
            
        return important_features

# Example usage
if __name__ == "__main__":
    # Sample texts
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Natural language processing helps computers understand human language",
        "Deep learning models require large amounts of training data",
    ]
    
    # Create vectorizer instance
    vectorizer = TextVectorizer()
    
    # Get 3D vectors
    vectors = vectorizer.vectorize_texts(texts)
    
    # Print results
    print("3D Vectors for each text:")
    for i, vector in enumerate(vectors):
        print(f"Text {i + 1}: {vector}")
    
    # Print important features for each dimension
    print("\nMost important features for each dimension:")
    important_features = vectorizer.get_feature_importance()
    for i, features in enumerate(important_features):
        print(f"\nDimension {i + 1}:")
        for word, weight in features:
            print(f"  {word}: {weight:.4f}")
