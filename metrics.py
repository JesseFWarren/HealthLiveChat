import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from src.retrieval import retrieve_relevant_chunks

# embedding model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# evaluation queries and categories
CATEGORIES = {
    "General NBA History & Rules": [
        "When was the NBA founded?", "What are the main differences between NBA and FIBA rules?",
        "How does the NBA draft lottery work?", "What is the significance of the NBA-ABA merger?",
        "Who was the first commissioner of the NBA?"
    ],
    "Players & Achievements": [
        "When was Larry Bird drafted and to what team?", "Who holds the record for the most assists?",
        "How many championships has Michael Jordan won?", "How many years was Wilt Chamberlain in the NBA?",
        "Who has the most triple-doubles in NBA history?", "Which player has won the most NBA MVP awards?",
        "How many championships did Magic Johnson win with the Lakers?", "What was Kobe Bryant’s highest-scoring game?",
        "Which player was known as 'The Admiral' in the NBA?"
    ],
    "Teams & Championships": [
        "What team has the most NBA titles?", "Which NBA team has the longest winning streak in history?",
        "How many championships have the San Antonio Spurs won?", "Who was the head coach when the Chicago Bulls won their six championships?",
        "What was the biggest trade in NBA history?", "Which team won the first-ever NBA championship?"
    ]
}

# ground truth answers
EVALUATION_QUERIES = {
    # General NBA History and Rules
    "When was the NBA founded?": ["The NBA was founded on June 6, 1946, as the Basketball Association of America (BAA) before merging with the National Basketball League (NBL) in 1949."],
    "What are the main differences between NBA and FIBA rules?": ["The NBA uses a 24-second shot clock, while FIBA uses a 14-second reset. NBA courts are larger, and defensive goaltending is enforced differently."],
    "How does the NBA draft lottery work?": ["The NBA draft lottery determines the order of the first 14 picks using a weighted system, with the worst teams having the highest chance of winning the top pick."],
    "What is the significance of the NBA-ABA merger?": ["The NBA-ABA merger in 1976 brought four ABA teams into the NBA: the San Antonio Spurs, Denver Nuggets, Indiana Pacers, and New York Nets."],
    "Who was the first commissioner of the NBA?": ["Maurice Podoloff was the first commissioner of the NBA, serving from 1946 to 1963."],

    # Players and Achievements
    "When was Larry Bird drafted and to what team?": ["Larry Bird was drafted by the Boston Celtics with the sixth overall pick in the 1978 NBA draft."],
    "Who holds the record for the most assists?": ["The record for most assists is held by John Stockton."],
    "How many championships has Michael Jordan won?": ["Michael Jordan has won six NBA championships."],
    "How many years was Wilt Chamberlain in the NBA?": ["Wilt Chamberlain was in the NBA for 14 Seasons."],
    "Who has the most triple-doubles in NBA history?": ["Russell Westbrook holds the record for most career triple-doubles in NBA history."],
    "Which player has won the most NBA MVP awards?": ["Kareem Abdul-Jabbar has won the most NBA MVP awards, with a total of six."],
    "How many championships did Magic Johnson win with the Lakers?": ["Magic Johnson won five NBA championships with the Los Angeles Lakers."],
    "What was Kobe Bryant’s highest-scoring game?": ["Kobe Bryant scored a career-high 81 points in a game against the Toronto Raptors in 2006."],
    "Which player was known as 'The Admiral' in the NBA?": ["David Robinson, who played for the San Antonio Spurs, was known as 'The Admiral' due to his time at the Naval Academy."],

    # Teams and Championships
    "What team has the most NBA titles?": ["The Boston Celtics possess the most overall NBA championships with 18."],
    "Which NBA team has the longest winning streak in history?": ["The Los Angeles Lakers hold the longest winning streak in NBA history, winning 33 consecutive games during the 1971-72 season."],
    "How many championships have the San Antonio Spurs won?": ["The San Antonio Spurs have won five NBA championships, with titles in 1999, 2003, 2005, 2007, and 2014."],
    "Who was the head coach when the Chicago Bulls won their six championships?": ["Phil Jackson was the head coach of the Chicago Bulls during their six NBA championships in the 1990s."],
    "What was the biggest trade in NBA history?": ["One of the biggest trades in NBA history was the trade that sent Kareem Abdul-Jabbar from the Milwaukee Bucks to the Los Angeles Lakers in 1975."],
    "Which team won the first-ever NBA championship?": ["The Philadelphia Warriors won the first NBA championship in 1947."]
}

def is_relevant(retrieved_chunks, ground_truths, threshold=0.7):
    retrieved_embeddings = model.encode(retrieved_chunks)
    ground_truth_embeddings = model.encode(ground_truths)

    similarity_scores = util.pytorch_cos_sim(ground_truth_embeddings, retrieved_embeddings)
    return (similarity_scores > threshold).any().item()

def compute_recall_at_k(query, ground_truths, k=3):
    retrieved_chunks = retrieve_relevant_chunks(query, k=k)
    return float(is_relevant(retrieved_chunks, ground_truths))

def compute_mrr(queries, k=3):
    reciprocal_ranks = []

    for query, ground_truths in queries.items():
        retrieved_chunks = retrieve_relevant_chunks(query, k=k)
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            if is_relevant([chunk], ground_truths):
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0)

    return np.mean(reciprocal_ranks)

if __name__ == "__main__":
    recall_scores = {}
    category_scores = {category: [] for category in CATEGORIES}

    for query, ground_truths in EVALUATION_QUERIES.items():
        recall_at_3 = compute_recall_at_k(query, ground_truths, k=3)
        recall_scores[query] = recall_at_3
        for category, questions in CATEGORIES.items():
            if query in questions:
                category_scores[category].append(recall_at_3)
        print(f"Query: {query}\nRecall@3: {recall_at_3:.2f}\n")

    avg_recall = np.mean(list(recall_scores.values()))
    mrr_score = compute_mrr(EVALUATION_QUERIES, k=3)

    print(f"\nAverage Recall@3: {avg_recall:.2f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr_score:.2f}")

    categories = list(CATEGORIES.keys())
    category_avg_scores = [np.mean(category_scores[cat]) for cat in categories]

    # Histogram and barchart to visualize
    plt.figure(figsize=(8, 5))
    plt.hist(list(recall_scores.values()), bins=np.arange(0, 1.2, 0.2), color='royalblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Recall@3 Score")
    plt.ylabel("Number of Queries")
    plt.title("Distribution of Recall@3 Scores")
    plt.xticks(np.arange(0, 1.2, 0.2))
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.bar(categories, category_avg_scores, color=['blue', 'green', 'red'], edgecolor='black', alpha=0.7)
    plt.xlabel("Question Category")
    plt.ylabel("Average Recall@3 Score")
    plt.title("Retrieval Performance Across Different NBA Categories")
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()