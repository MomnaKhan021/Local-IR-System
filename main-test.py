# homework assignment 03
# Momna Waryam Khan
#MSCS25011


import nltk
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Setting up logging to capture output in a file
logging.basicConfig(filename='ir_system_output.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Download necessary NLTK resources
nltk.download('punkt')  # Download punkt tokenizer
nltk.download('stopwords')  # Download stopwords

# Load the CSV file with error handling for different encodings
csv_file_path = 'Articles.csv'  # Ensure the correct path

try:
    # Try using 'ISO-8859-1' encoding
    data = pd.read_csv(csv_file_path, encoding='ISO-8859-1')
    logging.info("CSV file loaded successfully with 'ISO-8859-1' encoding.")
except UnicodeDecodeError as e:
    logging.error(f"UnicodeDecodeError with 'ISO-8859-1' encoding: {e}")
    print(f"Error loading CSV file with 'ISO-8859-1' encoding: {e}")
    try:
        # Try using 'utf-16' encoding
        data = pd.read_csv(csv_file_path, encoding='utf-16')
        logging.info("CSV file loaded successfully with 'utf-16' encoding.")
    except UnicodeDecodeError as e:
        logging.error(f"UnicodeDecodeError with 'utf-16' encoding: {e}")
        print(f"Error loading CSV file with 'utf-16' encoding: {e}")
        try:
            # Finally, try using 'utf-8-sig' encoding
            data = pd.read_csv(csv_file_path, encoding='utf-8-sig')
            logging.info("CSV file loaded successfully with 'utf-8-sig' encoding.")
        except Exception as e:
            logging.error(f"Error loading CSV file: {e}")
            print(f"Error loading CSV file: {e}")
            exit()

# Check the columns of the CSV file to find the correct column names
print(data.columns)  # To confirm the correct column names

# Assuming the correct columns are 'Article' for the document content and 'NewsType' for categories
try:
    documents = data['Article'].tolist()  # Using 'Article' for document content
    categories = data['NewsType'].tolist()  # Using 'NewsType' for category labels
    logging.info("Documents and categories loaded successfully.")
except KeyError as e:
    logging.error(f"Error: Missing expected columns in CSV: {e}")
    print(f"Error: Missing expected columns in CSV: {e}")
    exit()

# Log the first 5 documents
logging.info("First 5 Documents:")
logging.info(documents[:5])

# Preprocessing function
def preprocess(doc):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(doc.lower())  # Tokenization and lowercasing
    words = [word for word in words if word.isalnum()]  # Remove punctuation
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

# Preprocess the entire corpus
processed_corpus = [preprocess(doc) for doc in documents]

# TF-IDF Vectorization (Sparse Matrix)
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(processed_corpus)

# Query input
query = "What is the process of ranking documents based on relevance?"

# Preprocess the query
query_processed = preprocess(query)

# Convert query to TF-IDF vector
query_tfidf = vectorizer.transform([query_processed])

# Compute cosine similarity between query and documents (using sparse matrix)
cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix)

# Log the cosine similarities for each document
logging.info("\nCosine Similarities between Query and Documents:")
for idx, score in enumerate(cosine_similarities[0]):
    logging.info(f"Document {idx+1}: {score}")

# Rank documents based on cosine similarity
top_docs = cosine_similarities.flatten().argsort()[-3:][::-1]
logging.info("\nTop Documents Based on Cosine Similarity:")
for idx in top_docs:
    logging.info(f"Document {idx+1}: {documents[idx]} (Score: {cosine_similarities[0][idx]})")

# Assume we know the relevant documents for this query (example)
relevant_docs = [0, 2]  # Assume document 1 and document 3 are relevant

# Precision, Recall, and F1-Score calculation
retrieved_docs = top_docs.tolist()

# Handle Zero Division in Precision, Recall, and F1-Score
precision = len(set(retrieved_docs) & set(relevant_docs)) / len(retrieved_docs) if len(retrieved_docs) > 0 else 0
recall = len(set(retrieved_docs) & set(relevant_docs)) / len(relevant_docs) if len(relevant_docs) > 0 else 0

# F1-Score calculation
if precision + recall > 0:
    f1_score = 2 * (precision * recall) / (precision + recall)
else:
    f1_score = 0  # If both precision and recall are zero, F1 will be zero

logging.info(f"\nPrecision: {precision}")
logging.info(f"Recall: {recall}")
logging.info(f"F1-Score: {f1_score}")

# --- Integrating K-Means Clustering ---
# K-Means Clustering
num_clusters = 5  # Number of clusters to form
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(tfidf_matrix)

# Predict clusters for each document
predicted_clusters = kmeans.predict(tfidf_matrix)

# Log cluster information for each document
logging.info("\nDocument Clusters:")
for idx, cluster in enumerate(predicted_clusters):
    logging.info(f"Document {idx+1}: Cluster {cluster+1}")

# --- Integrating Naive Bayes Classification ---
# For simplicity, we'll use the same dataset to train and predict using Naive Bayes

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, categories, test_size=0.3, random_state=42)

# Train a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Predict the category for the query
query_class = nb_classifier.predict(query_tfidf)
logging.info(f"\nPredicted Category for Query: {query_class[0]}")

# Evaluate the Naive Bayes model
y_pred = nb_classifier.predict(X_test)

# Accuracy of Naive Bayes classifier
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"\nNaive Bayes Classification Accuracy: {accuracy}")

# --- Updating README.md ---
# Function to update the README file with the log data
def update_readme_with_log():
    try:
        # Read the content of the log file
        with open('ir_system_output.log', 'r') as log_file:
            log_data = log_file.read()

        # Open the README.md file in append mode
        with open('README.md', 'a') as readme_file:
            readme_file.write("\n### System Output\n")
            readme_file.write("The following is the output generated when running the IR system:\n")
            readme_file.write("```text\n")
            readme_file.write(log_data)  # Append the log data
            readme_file.write("```\n")
            print("Log data has been successfully appended to README.md")

    except Exception as e:
        print(f"Error while updating README.md: {e}")

# Call the function to update README.md with log data
update_readme_with_log()
