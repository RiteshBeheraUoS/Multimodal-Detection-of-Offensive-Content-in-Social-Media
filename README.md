# Multimodal Detection of Offensive Content in Social Media

**COMP6237: Data Mining, Coursework-1**

**Ritesh Kumar Behera**  
University of Southampton  
Student ID: 37614002  
Email: rkb1u25@soton.ac.uk

## Abstract

The proliferation of social media has resulted in an increase in the production and propagation of offensive material, which includes hate speech, cyberbullying, abusive messages, and toxic memes. It becomes quite difficult for traditional text-only detection methods to effectively classify offensive content because the offensive nature of the content lies both in the images and the corresponding text descriptions. The objective of this study is to design an offensive content detection system by making use of the multimodality of social media platforms. Data preprocessing, feature extraction, and advanced deep learning neural network techniques are employed in this approach to analyze meme images from social media platforms. Different data mining baselines are compared against advanced deep learning algorithms like transformer-based multimodal approaches.

## 1. Introduction

In addition to promoting faster interaction among people, the emergence of different types of social networking websites such as Facebook, Instagram, Twitter, and Reddit have led to the rapid diffusion of various forms of offensive and harmful materials such as:

- Hate speech
- Abuse
- Cyberbullying
- Racist comments
- Sexism
- Offensive memes

Detection of offensive material has been recognized as a critical problem in the field of data mining and machine learning due to the growing number of harmful messages available online.

### Early Detection Approaches

Earlier methods for detecting harmful online content primarily used textual data; however, in current social media posts, both images and text co-exist in a single message. Moreover, the offensive meaning of some social media messages is not easily identifiable without the combination of images and text. For instance, a meme message may be harmless without an adequate caption, but once added, the post could be perceived as offensive. The same case applies where an offensive text might be dependent on an image message. Therefore, the effective classification of such posts would require the use of both text and image data.

This study focuses on the development of a multimodal offensive content detection framework using data mining and deep learning methods.

### 1.1. Problem Statement

As of now, the main approaches of moderating social media include either human moderation or content analysis based on text only. However, none of these two methods is efficient in dealing with the growing amount of data available online:

- **Human moderation** involves inconsistency and takes an immense effort to perform
- **Text-based models** might struggle with identifying the offensive meaning present in the image and other visual components of multimedia content

**Project Objective:**

The main purpose of this project is to create a multimodal offensive content detection framework that would be capable of processing and analysing images that have some text captions embedded on them. The project aims to conduct a comparison between traditional data mining techniques and contemporary deep learning techniques to highlight the benefits and show improvements in the accuracy, precision, and F1-score.

The experimental setup employed in this project is documented within the GitHub repository referenced in this report.

### 1.2. Dataset

The project uses a multi-modal dataset from social media commonly known as 'memes'. This dataset was chosen because it includes images which introduce a layer of complexity where every meme is almost unique and can be interpreted differently depending on the person interacting with it.

**Dataset Characteristics:**
- Contains social media memes with different characters like hashtags, mentions, emojis, abbreviations, slang words and others
- Obtained from the Kaggle platform
- Pre-divided into training and testing dataset to prevent data bias and ensure fair evaluation

### 1.3. Evaluation Strategy

The project uses the classic statistical evaluation strategy of calculating accuracy, precision, and F1-score of the offensive content detected by models at different phases of the project. In order to address the scalability of the models employed in real-world applications, precision and F1-score are given importance, since the goal should be to correctly detect and prevent the propagation of as many offensive meme contents as possible.

## 2. Methodology

The project methodology follows a structured pipeline. The classification task begins with a feature extraction stage, where relevant attributes are derived from the raw image data. This is followed by the application of traditional data mining techniques to establish baseline performance and insights. The approach then evolves to incorporate deep learning techniques, employing neural network–based architectures to capture more complex patterns in the data.

### 2.1. Data Preprocessing and Feature Extraction

The feature extraction stage can be considered the most critical component of the project, as all subsequent phases depend heavily on its outputs. In this work, several pre-trained, commercially available neural network models are utilized to leverage transfer learning and extract diverse and informative features from the meme image dataset.

#### 2.1.1. Text Extraction

First, the text embeddings within meme images are extracted using Tesseract OCR via the pytesseract library. Each image is preprocessed to improve recognition accuracy by:

1. Converting to grayscale
2. Upscaling if its largest dimensions are below 800 pixels
3. Median-blurring for denoising
4. Histogram equalising for contrast normalisation
5. Binarising using Otsu thresholding
6. Automatically inverting dark-background images
7. Applying morphological opening step to remove residual noise

Tesseract model is then run on the preprocessed image using page segmentation mode 6 (uniform block of text) and OCR engine mode 3 (LSTM + legacy) and string outputs are obtained.

#### 2.1.2. Image Cleaning

To remove text overlays present in meme images, EasyOCR is used. It detects text regions within each image. Three cleaning methods are then applied:

**Method 1:** Draws a semi-transparent red overlay over detected text regions

**Method 2:** Uses OpenCV TELEA inpainting to fill detected text regions

**Method 3:** Uses DeepFillV2, a deep learning-based inpainting model, to reconstruct the underlying image content

The cleaned images are subsequently used as input for all following feature extraction steps, as the neural inpainting produces the most visually coherent results.

#### 2.1.3. Entity Label Extraction

Next, Object-level features are extracted from the cleaned meme images using a YOLOv8 model fine-tuned on the Open Images v7 taxonomy (yolov8m-oiv7.pt). For each image, the detector is run with a confidence threshold of 0.35, generating per-detection records comprising:

- Object class label
- Confidence score
- Bounding box coordinates (x₁, y₁, x₂, y₂)
- Derived width and height

This structured output enables downstream analysis of object co-occurrence patterns and their spatial relationships within meme images, providing semantic grounding for the classification task.

#### 2.1.4. Human Feature Extraction

In order to capture the human demographic features present in meme imagery, a two-stage detection and classification pipeline is constructed.

**Stage 1:** YOLOv8n (yolov8n.pt) is used to localise bounding boxes within each image, identifying all regions corresponding to person class.

**Stage 2:** Each detected crop is passed to a FairFace ResNet-34 classifier (res34_fair_align_multi_7_20190809.pt), a model pre-trained on a demographically balanced dataset to jointly predict race, gender, and age.

**Classifier Output:**

The classifier produces an 18-dimensional output vector, decomposed into three attribute groups:

- **Race (7 logits):** White, Black, Latino/Hispanic, East Asian, Southeast Asian, Indian, Middle Eastern
- **Gender (2 logits):** Male, Female
- **Age (9 logits):** 0–2, 3–9, 10–19, 20–29, 30–39, 40–49, 50–59, 60–69, 70+

Softmax probabilities are computed independently over each attribute group, and the argmax class together with its associated confidence score are recorded per detected face. FairFace's demographic outputs serve as a proxy signal for examining how demographic representation correlates with hateful content classification.

#### 2.1.5. Image Caption Generation

In order to derive a natural language semantic representation of each meme's visual content, the BLIP (Bootstrapped Language-Image Pre-training) model is applied to the full cleaned image dataset. Each image is processed through the BLIP feature extractor and conditional generation head, producing an unconditional free-text caption via greedy decoding.

This text description feature provides a semantic bridge between the visual content of a meme and text-based modelling, enabling downstream classifiers to jointly leverage image descriptions and the original meme caption text as complementary linguistic signals.

### 2.2. Phase-I Data Mining Baseline Modelling

The First Modelling phase of the project makes use of traditional Data Mining Techniques. These techniques are further categorised as Document Filtering Data Mining Techniques, and the use of Latent Semantic Space for classification purposes.

#### 2.2.1. Document Filtering Data Mining

For the document-filtering baseline, the task was formulated into three subsequent sub-tasks. Each sub-task employs the same pipeline and backbone classification architecture, but differs in the structure of the input. The core idea is to see how well traditional classifiers perform as more visual information is gradually fed into the model alongside the original text.

**Three Pipelines Tested:**

1. **Pipeline 1:** Relies solely on the meme's text caption (text-only baseline)
2. **Pipeline 2:** Adds FairFace demographic attributes (such as inferred race, gender, and age group) along with object-detection labels, all converted into structured tokens
3. **Pipeline 3:** Appends a generated natural-language description of the image, giving the model a richer understanding of the visual content

**Mathematical Formulation:**

The three inputs are defined as:

```
D⁽¹⁾ᵢ = Tᵢ
D⁽²⁾ᵢ = Tᵢ ⊕ Fᵢ ⊕ Oᵢ
D⁽³⁾ᵢ = Tᵢ ⊕ Fᵢ ⊕ Oᵢ ⊕ Cᵢ
```

where Tᵢ, Fᵢ, Oᵢ, and Cᵢ denote the caption, FairFace features, object labels, and image description respectively, and ⊕ denotes concatenation.

**Vectorisation Methods:**

Two vectorisation methods are used:

**Method 1: Binary Bag-of-Words**

```
xᵢⱼ = { 1, if token j appears in document i
       { 0, otherwise
```

**Method 2: TF-IDF Representation**

```
tfidfdᵢ,ⱼ = tfᵢ,ⱼ × log(N / dfⱼ)
```

where tfᵢ,ⱼ is the frequency of token j in document i, N is the total number of documents, and dfⱼ is the number of documents containing token j.

**Classifiers Evaluated:**

Four classifiers are evaluated across all three pipelines:

1. Naïve Bayes classifier built from scratch
2. Bernoulli Naïve Bayes on binary features
3. Multinomial Naïve Bayes on TF–IDF features
4. Fisher's method, which combines token-level probabilities into a single summary score

**Naïve Bayes Classification:**

The Naïve Bayes classifier predicts the class with the highest posterior probability:

```
ŷ = arg max P(c) ∏ P(xⱼ|c)
    c ∈ {0,1}   j=1ᵛ
```

where c is the class label, V is the vocabulary size, and xⱼ is the presence or absence of token j.

**Smoothing Technique:**

To avoid zero probabilities for unseen tokens:

```
P(wⱼ|c) = (Njc + α) / (Nc + αV)
```

where Njc is the number of times token j appears in class c, Nc is the total token count for class c, V is the vocabulary size, and α is the smoothing parameter.

The models are evaluated using accuracy, precision, and F1-score, with particular focus on the offensive class.

#### 2.2.2. Latent Semantic Space

The next phase-1 baseline uses data mining technique to identify the semantic patterns among the input textual data. Apart from it, it is important to identify or determine the predictive power of the inputs when projected into a latent semantic space. This technique focuses on the challenge of semantic mapping.

**Semantic Representation Challenge:**

The challenge is about mathematically representing documents so that semantically similar inputs are grouped together. To represent the meme captions, TF-IDF (Term Frequency-Inverse Document Frequency) is used. It processes weighted words based on their importance to a specific document relative to the entire corpus. It penalises highly common words, i.e., words that occur in every document.

**TF-IDF Formula:**

```
wᵢ,ⱼ = tfᵢ,ⱼ × log(N / dfᵢ)
```

**Latent Semantic Analysis:**

A core concept of Latent Semantic Analysis (LSA) is applied using Truncated Singular Value Decomposition (SVD). Truncated SVD is a low-rank approximation method that decomposes the term-document matrix into three parts (U, Σ, Vᵀ):

```
A ≈ Aₖ = UₖΣₖVᵀₖ
```

Since the raw input matrix is known to be noisy, SVD allows us to keep only the top k=100 singular values. This helps in reducing the dimensions and also preserving the most important semantic structures. The projection of both words and documents into a shared 100-dimensional coordinate system allows the synonyms or related terms to be properly aligned. It allows the model to perform discovery based on "concepts."

**Multimodal Extension:**

This LSA concept is then extended into a multimodal shared space through the concatenation of the TF-IDF vectors of the meme captions with the generated image description text. By concatenating, a "joint document" for each record comes into the picture. A latent space is achieved after using the truncated SVD on this combined matrix. In the latent space, visual descriptions and textual captions are projected into the same semantic embedding. In the final step, the logistic regression classifier is used on these latent coordinates to provide a binary classification.

**Logistic Regression:**

```
P(y = 1|x) = 1 / (1 + e^(-(β₀+β₁x)))
```

### 2.3. Phase-II Advanced Deep Neural Network Techniques

The second phase of the project makes use of more advanced deep machine learning tools to extract and use the features from both the text captions as well as the images from the meme image dataset. The project makes use of UNiversal Image-TExt Representation (UNITER), a large-scale pre-trained model for joint multimodal embedding.

**Model Architecture:**

UNITER adopts Transformer as the core backbone of the model to leverage its elegant self-attention mechanism designed for learning contextualised representations. The model architecture follows the Image-Text matching framework, which takes input features from:

- Cleaned image
- Extracted text captions
- Other metadata (object labels and facial data)

into a unified multimodal fusion. Unlike dual-stream architectures, the UNITER model follows a single-stream Transformer design that allows text tokens and image region tokens to attend to each other simultaneously through self-attention, enabling stronger cross-modal interaction and contextual understanding between textual and visual features.

**Processing Pipeline:**

1. **Visual Features:** Pre-trained ResNet-50 model extracts visual features from the image dataset into feature vectors, which are then projected into the hidden embedding space of the Transformer

2. **Textual Features:** Meme captions combined with generated object labels are tokenised and embedded using a BERT tokeniser and embedding layer

3. **Feature Fusion:** Combined features fed into the Transformer encoder to enable training across multiple self-attention layers

4. **Classification:** Pooled token representation from Transformer output layer is fused with encoded FairFace feature vectors before being fed into the Classifier head consisting of:
   - Fully connected layers
   - ReLU activation
   - Dropout regularization
   - Sigmoid-based binary classification layer

## 3. Results and Discussions

All classification tasks are conducted separately in subsequent steps, and the observations are recorded for evaluation purposes.

### 3.1. Phase-I Data Mining Baseline Modelling

#### 3.1.1. Document Filtering Results

The document-filtering experiments compare three input pipelines using the same classifier structure. The first pipeline uses only text captions, the second adds FairFace features and object labels, and the third adds generated image descriptions in addition to the previous features. This allows the effect of each additional input type to be compared directly.

| Pipeline | Model | Accuracy | Precision | F1-score |
|----------|-------|----------|-----------|----------|
| Text only | Naïve Bayes | 55.4% | 57.3% | 48.7% |
| Text only | BernoulliNB | 52.0% | 56.8% | 25.9% |
| Text only | MultinomialNB | 50.2% | 52.0% | 09.5% |
| Text only | Fisher | 54.8% | 53.8% | 60.1% |
| Text + FairFace + Objects | Naïve Bayes | 55.6% | 55.4% | 56.3% |
| Text + FairFace + Objects | BernoulliNB | 53.0% | 60.3% | 27.2% |
| Text + FairFace + Objects | MultinomialNB | 51.2% | 66.7% | 09.0% |
| Text + FairFace + Objects | Fisher | 53.8% | 52.8% | 60.6% |
| Text + FairFace + Objects + Desc. | Naïve Bayes | 57.6% | 56.1% | 62.1% |
| Text + FairFace + Objects + Desc. | BernoulliNB | 53.6% | 60.0% | 31.8% |
| Text + FairFace + Objects + Desc. | MultinomialNB | 49.6% | 40.0% | 03.1% |
| Text + FairFace + Objects + Desc. | Fisher | 53.0% | 52.4% | 58.7% |

**Table 1:** Comparison of document-filtering models across the three input pipelines.

**Key Observations:**

As observed from Table 1, the first text-only pipeline provides a baseline for measuring how much offensive-content information is present in the meme captions alone. Fisher's method achieves the best result in this setting, with an F1-score of 60.1%. This shows that captions contain useful offensive-content signals, but the overall accuracy remains limited because text alone often misses the broader visual context of a meme.

In the second pipeline, FairFace demographic tokens and object-detection labels are added to the captions. Fisher's method again performs best, achieving an F1-score of 60.6%. This is the highest recall across all three pipelines, suggesting that structured visual-context tokens help the model detect more offensive examples.

The third pipeline adds generated image descriptions to the text, FairFace labels, and object labels. This produces the best overall result. The scratch-built Naïve Bayes model achieves the highest accuracy of 57.6% and the highest F1-score of 62.1%. This suggests that the generated descriptions provide additional natural-language context about the image, helping the model balance offensive content detection with fewer incorrect predictions.

**Performance Improvements:**

The best F1-score improves from 60.1% in the text-only pipeline to 60.6% when FairFace and object labels are added, and then to 62.1% when generated image descriptions are included. This indicates that additional visual-context information improves the document-filtering baseline.

**Classifier Comparison:**

The comparison also shows that Multinomial Naïve Bayes with TF–IDF is not well suited to this task. Its F1-score remains very low across all three pipelines, which suggests that weighted token frequency is less useful than binary token presence for this dataset. In contrast, Fisher's method and the scratch Naïve Bayes model perform more consistently. Fisher's method is strongest when F1-score is prioritised, while the scratch Naïve Bayes model achieves the best overall balance in the full multimodal pipeline.

#### 3.1.2. Latent Semantic Analysis

The phase 1 analysis helps in evaluating Latent Semantic Analysis (LSA) as well. It tells how the semantic structure of the meme dataset is captured. Patterns can be observed by transforming raw text into a dense latent space.

**Cumulative Explained Variance:**

The scree plot shows cumulative explained variance across the hundred singular values. The principle of low-rank approximation clearly confirms that the dataset shows a steady or linear accumulation of information. Based on this information, it is clear that the semantic patterns in the memes are very diverse and distributed or spread across many concepts. By choosing the value of k as 100, all the important information required to represent the core concepts of the memes is retained. This technique also filters and reduces the extreme sparsity of the initial TF-IDF matrix.

**Latent Concepts Analysis:**

The bar charts provide information about the top positive words that define latent dimensions. There are two models: the text-only model and the multimodal expansion model. In the text-only model, the primary concepts are dependent on high-frequency discriminative words like "Hitler," "character," and "love." In the multimodal expansion model, the concepts shift to visual descriptors (e.g., "sitting," "man," and "standing"). This shift clearly confirms that truncated SVD is successfully projecting textual captions and image descriptions into a shared semantic coordinate system.

**Cosine Similarity Analysis:**

The cosine similarity heatmap provides information about the semantic closeness between different memes. A high score means memes are grouped together in the latent space, and a low score means the opposite. This clearly confirms that the model is able to find patterns based on the angle between the vectors.

**LSA Performance Summary:**

| Experiment Phase | Accuracy | Precision | F1-Score |
|------------------|----------|-----------|----------|
| Phase-I: Text-only LSA | 53.60% | 68.0% | 23.0% |
| Phase-II: Multimodal LSA | 54.20% | 73.0% | 22.0% |

**Table 2:** Overall Performance Comparison for LSA

### 3.2. Phase-II Advanced Deep Neural Network Techniques

| Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|----------|-----------|--------|----------|---------|
| 64.8% | 70.33% | 51.20% | 59.26% | 71.05% |

**Table 3:** Evaluation results for UNITER-based DL model

As can be observed from Table 3, the advanced neural network-based classifier model achieves higher success in detecting the presence of offensive content in a meme image. The multimodal UNITER-based architecture seems to learn more effectively than traditional methods the semantic relationships between textual captions and visual image features present in the meme image through joint embedding and Transformer based attention mechanism.

**Challenges and Limitations:**

Nevertheless, there remains significant improvement that the architecture fails to achieve. This is largely due to:

1. **High Complexity and Ambiguity:** Unlike conventional image classification tasks, memes require a large context to be fully interpreted. These can include hidden cultural references, implicit content, significant historical events, and text-to-image relations that can be very difficult for machine learning models to understand.

2. **Limited Dataset Size and Quality:** Transformer-based multimodal architectures such as UNITER require extremely large and diverse datasets to learn high-quality data embeddings that map between different data types, such as text and images. Since the dataset used in this project is comparatively limited, the model may not have been exposed to sufficient variations of offensive meme patterns, thereby limiting its generalisation capability.

3. **Uniqueness of Memes:** Almost every meme is usually unique, which also makes it even more difficult for the model to develop a pattern to use in the detection of offensive content.

4. **Dataset Imbalance:** Dataset imbalance between offensive and non-offensive samples may also have contributed towards the reduced recall performance.

5. **Visual Feature Extraction Limitations:** Although ResNet-50 provides strong image representations, it may fail to capture certain cues, gestures, or even objects that are commonly present in offensive memes. Similarly, OCR inaccuracies can also negatively impact the textual understanding capacity of the model.

**Path Forward:**

Therefore, although the proposed multimodal Transformer architecture demonstrates promising performance and validates the effectiveness of joint visual-textual learning, achieving state-of-the-art offensive meme detection remains a challenging task. Future improvements could move in the direction of:

- Integrating knowledge in form of knowledge graphs and text documents which could help machine learning models get a better understanding of meme contexts
- Training on bigger and diversified meme datasets to improve performance and generalization capabilities

## 4. Conclusion

The results demonstrate that both traditional document-filtering techniques and advanced deep learning–based multimodal architectures contribute valuable insights into offensive meme detection, while also highlighting the increasing importance of multimodal learning in modern data mining applications.

**Key Findings:**

1. **Traditional Data Mining Foundation:** The traditional data mining techniques established a strong analytical foundation producing structured representations that could be evaluated using classical machine learning algorithms. Although the baseline models achieved moderate performance, they revealed important semantic relationships within the dataset and demonstrated that offensive content can be partially detected through statistical and vector-space modeling techniques.

2. **Advanced Deep Learning:** Building on this foundation, the advanced multimodal deep learning techniques were capable of jointly learning relationships between textual captions and visual image features through shared embeddings and attention mechanisms, which showed great improvement in offensive-content detection performance. This demonstrates how modern deep learning approaches extend and refine the principles established by traditional data mining techniques rather than replacing them entirely.

3. **Remaining Challenges:** However, the findings also indicate that advanced data mining techniques are not yet fully sufficient for reliable offensive meme detection. Despite achieving a stronger overall classification performance, the deep learning–based approach still struggled with issues such as contextual ambiguity, cultural references and implicit offensive meaning.

**Future Research:**

Further research is required to improve robustness, contextual awareness and generalisation performance. Future work could consider knowledge-enhanced learning using knowledge graphs and text documents to better understand implicit and culturally dependent offensive content.

## References

[1] Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. 2020. UNITER: Universal Image-Text Representation Learning. [Online]. Available: https://arxiv.org/abs/1909.11740

[2] COMP6237. 2026. Multimodal-Detection-of-Offensive-Content-in-Social-Media. [Online]. Available: https://github.com/RiteshBeheraUoS/Multimodal-Detection-of-Offensive-Content-in-Social-Media

[3] Aijing Gao, Bingjun Wang, Jiaqi Yin, and Yating Tian. 2021. Hateful Memes Challenge: An Enhanced Multimodal Framework. [Online]. Available: https://arxiv.org/abs/2112.11244

[4] Samuel Hoffstaetter. Python Tesseract: pytesseract. [Online]. Available: https://pypi.org/project/pytesseract/

[5] Zhiwu Huang. 07 Document Filtering Spam Ham. [Online]. Available: https://github.com/zhiwu-huang/COMP6237-Data-Mining-Demo-Code/blob/master/07_document_filtering_spam_ham.ipynb

[6] Zhiwu Huang. 11 Semantic Spaces Visual Demo. [Online]. Available: https://github.com/zhiwu-huang/COMP6237-Data-Mining-Demo-Code/blob/master/11_semantic_spaces_visual_demo.ipynb

[7] Glenn Jocher, Jing Qiu, and Ayush Chaurasia. Ultralytics YOLO Version 8.0.0, January 2023. [Online]. Available: https://github.com/ultralytics/ultralytics

[8] Kaggle. Facebook Hateful Meme Dataset. [Online]. Available: https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset/data

[9] Kimmo Karkkainen and Jungseock Joo. 2021. FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 1548–1558.

[10] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022. BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation. [Online]. Available: https://arxiv.org/abs/2201.12086

[11] Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas Huang. 2019. Free-form Image Inpainting with Gated Convolution. [Online]. Available: https://arxiv.org/abs/1806.03589
