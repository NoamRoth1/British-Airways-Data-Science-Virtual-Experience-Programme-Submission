# British Airways Data Science Virtual Experience Programme Submission

This repository contains my British Airways Data Science Virtual Experience Programme submission. The project focuses on analyzing flight reviews for British Airways using data science techniques. The goal is to gain valuable insights from customer feedback to provide future recommendations for improving services.

## Project Overview
The project consists of the following components:

1. Data Collection:
   - The web scraping technique was used to collect flight reviews from the "https://www.airlinequality.com/airline-reviews/british-airways" website.
   - BeautifulSoup and requests libraries were employed to extract the reviews.
   - A total of 10 pages, with 100 reviews per page, were scraped to obtain a substantial dataset.

2. Data Preprocessing:
   - The collected reviews underwent preprocessing to remove certain elements such as "✅ Trip Verified," "❎," and "Not Verified" for cleaner text data.
   - The "|" character was also eliminated from the reviews to enhance readability.

3. Sentiment Analysis:
   - The SentimentIntensityAnalyzer from the NLTK library was utilized for sentiment analysis.
   - Each review was processed, and sentiment scores were generated using the polarity_scores() function.
   - Based on the compound score, a sentiment label ("Positive" or "Negative") was assigned to each review.

4. Sentiment Distribution Visualization:
   - The project includes a bar chart visualization to showcase the sentiment distribution of the flight reviews.
   - The chart was created using the matplotlib library, with the x-axis representing sentiment categories and the y-axis representing the count of reviews for each sentiment.

## Repository Structure

- `data/`: This directory contains the dataset files generated during the project.
   - `BA_reviews.csv`: The processed dataset containing flight reviews and their corresponding sentiments.

- `BA_sentiment_analysis.ipynb`: Jupyter Notebook containing the code for data collection, preprocessing, sentiment analysis, and sentiment distribution visualization.
- `BA bar chart.png`: The bar chart image illustrating the sentiment distribution of the flight reviews.

## Getting Started
To explore and run the project, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/NoamRoth1/British-Airways-Data-Science-Virtual-Experience-Programme-Submission.git
```

2. Install the required dependencies. It is recommended to use a virtual environment:

```bash
cd British-Airways-Data-Science-Virtual-Experience-Programme-Submission
pip install -r requirements.txt
```

3. Open and run the `BA_sentiment_analysis.ipynb` Jupyter Notebook. This notebook contains the code for data collection, preprocessing, sentiment analysis, and sentiment distribution visualization.

4. The notebook will generate the processed dataset (`BA_reviews.csv`) and the sentiment distribution bar chart (`BA bar chart.png`) in the `data/` directory.


## Conclusion
The British Airways Data Science Virtual Experience Programme Submission provides an example of how data science techniques can be applied to analyze customer feedback and extract valuable insights. The sentiment analysis and visualization highlight the sentiment distribution, allowing for a better understanding of customer opinions. The project can be expanded further to include more advanced analyses or integrated into a larger data-driven decision-making process for British Airways.
