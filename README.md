<h1>Spam Email/SMS Classifier</h1>

<h3>Description</h3>
<p>This project focuses on building a machine learning model to classify emails or SMS as spam or not spam. The solution uses natural language processing (NLP) techniques and a Naive Bayes classifier for accurate predictions. The classifier is deployed as a web application using Streamlit.</p>

<h3>Features</h3>
<ul>
  <li><strong>Data Preprocessing</strong>: Tokenization, stopword removal, and stemming of email/SMS content.</li>
  <li><strong>Feature Extraction</strong>: Utilizes TF-IDF vectorization for transforming text data into numerical features.</li>
  <li><strong>Spam Classification</strong>: Implements a Multinomial Naive Bayes classifier for spam detection.</li>
  <li><strong>Model Deployment</strong>: Provides a user-friendly web interface for classification using Streamlit.</li>
  <li><strong>Model Persistence</strong>: Saves the trained model and vectorizer for future use with pickle.</li>
</ul>

<h3>Technologies Used</h3>
<ul>
  <li><strong>Python 3.x</strong></li>
  <li><strong>Pandas</strong>: For data manipulation and analysis.</li>
  <li><strong>NumPy</strong>: For numerical operations.</li>
  <li><strong>NLTK</strong>: For natural language processing tasks (tokenization, stemming, stopword removal).</li>
  <li><strong>Scikit-learn</strong>: For feature extraction (TF-IDF), machine learning (Naive Bayes), and evaluation.</li>
  <li><strong>Streamlit</strong>: For deploying the web-based classification app.</li>
  <li><strong>Pickle</strong>: For saving and loading the trained model and vectorizer.</li>
</ul>

<h3>Installation and Setup</h3>

<h4>Prerequisites</h4>
<p>Before running the project, ensure you have the following installed:</p>
<ul>
  <li>Python 3.x</li>
  <li>pip (Python package manager)</li>
</ul>

<h4>Steps to Run the Application</h4>
<ol>
  <li><strong>Clone the repository</strong>:
    <pre><code>git clone https://github.com/your-username/spam-classifier.git
cd spam-classifier</code></pre>
  </li>
  <li><strong>Set up a virtual environment (optional but recommended)</strong>:
    <pre><code>python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate</code></pre>
  </li>
  <li><strong>Install required dependencies</strong>:
    <pre><code>pip install -r requirements.txt</code></pre>
  </li>
  <li><strong>Run the Streamlit app</strong>:
    <pre><code>streamlit run app.py</code></pre>
    The application will launch in your default web browser.
  </li>
</ol>

<h3>File Structure</h3>
<pre><code>spam-classifier/
|-- model.py
|-- app.py
|-- spam.csv
|-- spam_classifier_model.pkl
|-- vectorizer.pkl
|-- requirements.txt
|-- README.md</code></pre>

<h4>File Descriptions:</h4>
<ul>
  <li><strong>model.py</strong>: Contains code for preprocessing, training the classifier, and saving the model/vectorizer.</li>
  <li><strong>app.py</strong>: The Streamlit application for user interaction and classification.</li>
  <li><strong>spam.csv</strong>: The dataset used for training the model.</li>
  <li><strong>spam_classifier_model.pkl</strong>: The saved trained model.</li>
  <li><strong>vectorizer.pkl</strong>: The saved TF-IDF vectorizer.</li>
</ul>

<h3>How to Use</h3>
<ul>
  <li>Enter the content of an email or SMS into the input box.</li>
  <li>Click on the "Classify" button to determine if the input is spam or not.</li>
  <li>View the result displayed on the screen.</li>
</ul>

<h3>In Development</h3>
<p>This project is still under development, and the following enhancements are planned:</p>
<ul>
  <li>Improving the preprocessing pipeline for better handling of complex text data.</li>
  <li>Adding support for multilingual spam detection.</li>
  <li>Implementing additional machine learning models for comparison.</li>
</ul>

<h3>Future Enhancements</h3>
<ul>
  <li>Integrating deep learning models for more robust classification.</li>
  <li>Adding real-time email or SMS integration for live spam detection.</li>
  <li>Creating a mobile-friendly version of the web app.</li>
</ul>

<h3>Acknowledgments</h3>
<ul>
  <li><a href="https://www.nltk.org/">NLTK Documentation</a></li>
  <li><a href="https://scikit-learn.org/">Scikit-learn Documentation</a></li>
  <li><a href="https://streamlit.io/">Streamlit Documentation</a></li>
</ul>

<h3>Contributing</h3>
<p>Feel free to fork the repository, make changes, and submit pull requests. Contributions are always welcome!</p>
