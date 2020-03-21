# Natural-Language-Processing

This project includes different domains of Natural Language Processing:

1. Topic Modeling using GENSIM and GUIDED LDA

2. Classification of text into more than 2 classes.

3. Scraping of websites through their HTML tags using Selenium

4. Different Data Preprocessing Techniques.

5. Named Entity Recognition using SPACY


Topic Modeling:

Its a difficult task as it is unsupervised. We utilize the concept of LDA to do this. 
LDA is used to classify text in a document to a particular topic. 
It builds a topic per document model and words per topic model, modeled as Dirichlet distributions. 
Each document is modeled as a multinomial distribution of topics 
and each topic is modeled as a multinomial distribution of words.

To solve the issue of accurately identifying topics, I utilised the concept of seed words.
I referred to the Paper by Jagadeesh Jagarlamudi, Hal Daume III and Raghavendra Udupa Incorporating Lexical Priors into Topic Models. 
The paper talks about how the priors (in this case priors mean seeded words) can be set into the model to guide it in a certain direction.


Classification of Text based on polarity:

We have various state of the art models to do text classification. 
Challenge was to identify suitable dataset which would increase our test accuracy 
and should be similar to our end results. 
Utilized 10 GB airline sentiments dataset which has texts segregated into POSITIVE, NEGATIVE AND NEUTRAL.
Used VADER, GLOVE AND word embeddings through BERT which gave an accuracy of 90%.



