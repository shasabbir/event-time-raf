## TimeRAF: Retrieval-Augmented Foundation Model for Zero-Shot Time Series Forecasting

Huanyu Zhang , Chang Xu , Yi-Fan Zhang , Zhang Zhang , Member, IEEE, Liang Wang , Fellow, IEEE, [URL 🔗](https://orcid.org/0000-0002-8281-2314)

and Jiang Bian

Abstract—Time series forecasting plays a crucial role across numerous domains, driving rapid development in the field. With the advent of large models, time series foundation models (TSFMs) have exhibited great generalization capabilities, such as zero-shot learning, through large-scale pre-training. Meanwhile, Retrieval- AugmentedGeneration (RAG) methods are widely employed to en- hance the performance of foundation models on unseen data across various domains, including Large Language Models (LLMs). To explore the integration of TSFMs with retrieval-augmented meth- ods, we introduce TimeRAF, a Retrieval-Augmented Foundation model for zero shot time series Forecasting. A learnable retriever is employed and trained in an end-to-end fashion to extract useful in- formation froma curated time series knowledge base. Additionally, we propose an approach called Channel Prompting for knowledge integration. Augmented by the retrieved knowledge, our TimeRAF demonstrates significant improvement across various domain and datasets. Furthermore, TimeRAF can leverage specialized knowl- edge bases to meet diverse application requirements. Extensive ablation studies and visualizations are provided to validate the effectiveness of our approach.

Index Terms—Foundation Models, retrieval augmented

generation.

## I. INTRODUCTION

T IME series (TS) data is prevalent in applications like fore- casting [1], classification [2], generation [3], and anomaly detection [4]. Time series forecasting, in particular, has gained significant popularity in recent years due to its vital role in vari- ous domains, including finance [5], healthcare [6], weather [7], and traffic [8]. The popular approach in the past typically learns from single-domain, small-scale datasets [9], [10], which in- herently constrains their generalization capabilities. However, the landscape of time series analysis is evolving rapidly with the advent of large models. Time series foundation models (TSFMs), trained on large-scale, multi-domain datasets, have demonstrated zero-shot learning abilities, revolutionizing vari- ous time series domains and diverse applications [11], [12], [13]. [URL 🔗](#page-0)

Received 17December 2024; revised 6May 2025; accepted 9 June 2025. Date ofpublication 12 June 2025; date ofcurrent version 24 July 2025. Recommended for acceptance by H. Hu. (Corresponding author: Chang Xu.)

Huanyu Zhang, Yi-Fan Zhang, Zhang Zhang, and Liang Wang are with the State Key Laboratory of Multimodal Artificial Intelligence Systems (MAIS), Institute of Automation, Chinese Academy of Sciences (CASIA), Beijing 100190, China, and also with the School of Artificial Intelligence, University of Chinese Academy of Sciences (UCAS), Beijing 100049, China.

Chang Xu and Jiang Bian are with the Microsoft Research Asia, Beijing 100080, China (e-mail: chanx@microsoft.com). [URL 🔗](mailto:chanx@microsoft.com)

Digital Object Identifier 10.1109/TKDE.2025.3579137

1041-4347 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and similar technologies.

Personal use is permitted, but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

Meanwhile, Retrieval-Augmented Generation (RAG) is an increasingly prevalent technique that enhances the capabilities of foundation models in various domains, including text gen- eration [14] and image generation [15]. This approach allows models to access external knowledge through various infor- mation retrieval techniques, enabling them to gather supple- mentary information during the generation process. Typically, the retrieval knowledge can be sourced from external datasets in the same format with the training corpus. For instance, in dialogue systems, RAG can help generate more contextually relevant responses by retrieving previous dialogues or similar interactions from a database [16]. In the context of time series forecasting, however, the application of retrieval augmentation remains largely underexplored. This underscores a significant gap, considering that time series data is inherently dynamic, heterogeneous, and context-dependent, with forecasting perfor- mance often highly sensitive to distribution shifts and unseen domains. A natural question arises: Can the integration oftime series foundation models with retrieval-augmented techniques improve forecasting performance, particularly in zero-shot scenarioswhere labeled data is unavailable and generalization is critical? [URL 🔗](#page-0)

As an intuitive example, a pre-trained model trained on a

general time series dataset may struggle when forecasting for specific domains such as weather patterns in a particular re- gion, which is illustrated in the left plot of Fig. 1. However, by accessing knowledge bases, the model could dynamically retrieve relevant information—such as time series data from similarweather conditions—without requiring extensive param- eter updates. This allows the model to integrate relevant prior knowledge, improving its zero-shot forecasting capability. In this manner, retrieved external data provides valuable context and serves as an additional source of prior information, enabling more accurate predictions. These advancements motivate our ex- ploration ofRetrieval-Augmentation for time series Forecasting (RAF). However, designing an effective RAF framework for time series forecasting involves several key challenges: (1) What types of data can serve as knowledge bases to support time series models? (2) How can relevant knowledge be retrieved when encountering inputs from diverse domains? (3) How can retrieved knowledge be effectively integrated to improve model performance? [URL 🔗](#page-0)

To address these challenges, we introduce TimeRAF, a novel framework designed to leverage retrieval-augmented generation techniques for time series foundation models. As shown in the


*Fig. 1. Left: Time series foundation models (TSFMs), while capable of zero-shot forecasting, are limited by insufficient prior knowledge, resulting in constrained prediction accuracy. Right: By dynamically retrieving relevant information from a knowledge base, our TimeRAF enhances prediction accuracy, leading to more precise zero-shot forecasting performance.*

right of Fig. 1, by retrieving and integrating external time series data, we aim to overcome the limitations of existing TSFMs and enhance zero-shot time series forecasting performance. TimeRAF consists of a retriever that scores and selects relevant time series data from a knowledge base. The knowledge base can either be a comprehensive database composed of multiple datasets across various domains or a domain-specific database comprising a singular dataset relevant to test data. Furthermore, an end-to-end learnable retrieval methodology is introduced to ensure that the retrieved data delivers enhancement. To leverage retrieved time series, we introduce an effective approach, named Channel Prompting, to integrate the knowledge from retrieved data. Our extensive experiments on various datasets demonstrate that TimeRAF significantly achieves a substantial improvement over TSFM and outperforms several existing zero-shot time series forecasting methods. [URL 🔗](#page-0)

Overall, our contributions can be summarized as follows:

- r We propose TimeRAF, a novel framework that leverages retrieval augmentation techniques to enhance zero-shot time series forecasting. By retrieving relevant data from a knowledge base and effectively integrating the retrieved in- formation, TimeRAF supplements the pre-trained knowl- edge of foundation models, enhancing their forecasting capabilities.

- r We employ a learnable retriever to calculate retrieval scores for time serieswithin the knowledge base and select the best options. To integrate retrieved knowledge, we introduce Channel Prompting to extract valuable information from the retrieved data effectively.

- r Our TimeRAF demonstrates significant improvement on zero-shot forecasting through the incorporation of RAF into TSFM and outperforms several foundation models. Furthermore, we present comprehensive ablation studies and visualizations to evaluate the efficacy of our approach.

## II. RELATED WORK

Foundation Models for Zero-shot Time Series Forecasting: Recent years have witnessed the rise of TSFMs. TimeGPT- 1 [17] is the first closed-source model offering zero-shot forecasting capabilities. Lag-llama [18] leveraged the LLaMA [URL 🔗](#page-0)

architecture [19] with lagged time series features for time series forecasting. TimesFM [20] is a patch-based, decoder-only foun- dational model designed for time series forecasting, which em- ploys a larger output patch size to enhance decoding efficiency. The model is pre-trained on a comprehensive dataset sourced from Google Trends and Wikipedia pageviews, in combination with open data. MOIRAI [12] introduces LOTSA, a large-scale time series datasets, and utilizes it to train a foundation model based on a masked encoder architecture, which achieves compet- itive or superior performance as a zero-shot forecaster compar- [URL 🔗](#page-0)

ing to full-shot models. Timer

[[13]](#page-0)

and TimerXL

[[21]](#page-0)

use causal

Transformer framework for time series forecasting, achieving great zero-shot performance. UniTime [22] introduces a uni- fied model that leverages domain instructions and a language– time series transformer to support zero-shot and cross-domain time series forecasting. Tiny Time Mixers (TTMs) [23] lever- ages a lightweight mixer-style architecture and demonstrated remarkable zero-shot forecasting performance. Since TSFMs have shown potential in zero-shot time series forecasting, our approach aims to enha nce their generalization capabilities by applying RAG techniques to leverage external knowledge. [URL 🔗](#page-0)

Retrieval Augmented Generation for Foundation Models: Foundation Models like LLMs have achieved remarkable success, though they still face limitations in domain-specific or knowledge-intensive tasks. To address these challenges, various RAG methods have been proposed: DocPrompting [24] curated a retrieval annotation dataset to train a retriever for augmenting input in code generation. DPR [14] develops a dense embedding model for indexing passages in a low-dimensional, continuous space. RePlug [25] refined the retriever by distilling the knowledge from the language model’s probability. LAPDOG [16] introduces an end-to-end dense retriever framework specifically for personalized dialogue generation, emphasizing objective optimization. Beyond NLP tasks, RAG has also been applied to other domains: REACT [26] freezes the original model and updates only the additional trainable weights on the retrieved knowledge, significantly enhancing visual model’s zero-shot performance. Re-Imagen [15] uses retrieved information to produce high-fidelity and faithful images, even for rare or unseen entities. Additionally, in time series analysis, [URL 🔗](#page-0)


*Fig. 2. Overview of TimeRAF: TimeRAF utilizes a retriever to dynamically retrieve relevant candidates from a knowledge base and then utilizes the proposed Channel Prompting module to integrate knowledge between the retrieved data and the input. The knowledge-enhanced embeddings are subsequently fed into the backbone of the foundation model to improve forecasting results. During training, the backbone remains frozen.*

RATSF [27] develops a cross-attention module to integrate additional data for better prediction. But its retrieval process is constrained to historical data. ReTime [28] retrieves relational references to improve forecasting and imputation for incomplete time series. RAF [29] develops a retrieval-augmented-forecasting framework by concatenating the retrieved data with the query. TSGAssist [30] introduces an assistant that combines TSGBench with LLMs and RAG to support time series generation understanding, benchmarking, and recommendation. RATD [31] enhances time series diffusion forecasting via retrieval-augmented reference guidance during denoising. However, current methods lack a robust framework for integrating RAG with TSFMs in a zero-shot forecasting setting. In contrast, our work is specifically designed to address this gap. We use extensive public time series data to build a knowledge base and enhance zero-shot prediction in TSFMs with an effective RAF method. [URL 🔗](#page-0)

## III. METHOD

## A. Overview

An illustration of our TimeRAF framework is provided in Fig. 2. Firstly, a retriever is utilized for learning to retrieve rele- vant data from the knowledge base (refer to Section III-E). Fol- lowing this, the proposed Channel Prompting approach is em- ployed for the integration of retrieved knowledge. Therefore, the [URL 🔗](#page-0)

entire forecaster F is capable of harnessing external knowledge, thereby facilitating knowledge enhanced forecasting (refer to Section III-F). During training, the backbone of TSFM remains frozen. Details of training and inference process are provided in Sections III-E2 and III-G. Besides, the knowledge bases utilized for training and inference are detailed in Section III-D. [URL 🔗](#page-0)

## B. Problem Formulation

Following previous work [9], we employ the channel inpen- [URL 🔗](#page-0)

dent strategy. Let X ∈ Rsl×c be a multivariate time series of length sl and number of channels c. The input can be denoted

*TABLE I*

*SYMBOLS AND NOTATIONS*

| Symbols | Notations |
| --- | --- |
| Xx | Multivariate input |
| x | Univariate input |
|   | Input embedding of Channel Prompting |
|   | Input embedding of backbone model |
| Y | Ground truth |
| Y | Multivariate prediction |
| 9 | Univariate prediction |
| t | Entities in the knowledge base |
| c | Retrieved candidates |
| cue | Augmented retrieved candidates |
| ci | The ith retrieved candidate |
| & | Embedding of the ith retrieved candidate |
| Ss, | The ith augmented retrieved candidate |
|   | Top-k retrieval scores for query q |
| Ch | Augmented retrieval scores for q |
| P | Contribution probability |
|   | Augmented contribution probability |
| pi | Contribution probability of ¢; |
| Tim, Ts | Temperature hyperparameter |
| z | Concatenated embedding |
| Zz | Fused embedding |
|   | Forecaster |

as x ∈ Rsl×1, and the forecasting task can be formally defined

as predicting the future values ˆy ∈ Rfl×1 given the input. Upon completing the analysis of all data channels, the final compre-

hensive prediction result ˆY ∈ Rfl×c is derived. Here, fl denotes the forecast length/horizon. The ground truth is denoted as

Y ∈ Rfl×c. Given a set of retrieved time series data C from the knowledge base (details will be elaborated in Section III-E), we aim to leverage the valuable information within them to enhance the forecasting capability of the forecaster F. The entire process [URL 🔗](#page-0)

can be formulated as Yˆ = F(X,C). The important symbols and notations used in this paper are summarized in Table I. [URL 🔗](#page-0)


## C. Data Preprocessing

To ensure the quality and consistency of the time series data, we performed a series of data preprocessing steps following recentTSFMs[13], [23]. Tomitigate the effects ofvarying scales across different features, we applied Z-score normalization to all input sequences as well as retrieved sequences. This method transforms the data to have a mean of zero and a standard devia- tion ofone. Additionally, we applied a sliding window technique to create input-output pairs, where each window represents a sequence ofpast observations used to predict the next windowof future values. Subsequently, the input and retrieved data undergo preprocessing specific to the backbone model, such as patching. [URL 🔗](#page-0)

## D. Knowledge Base

In order to facilitate knowledge retrieval, it is essential to first establish a knowledge base. To enhance the efficiency of knowl- edge integration and extraction, we undertake a preprocessing of all sequences within the knowledge base to align with the dimensions of the lookback window, resulting in the following

representation: Knowledge Base = {ti|ti ∈ Rsl×1}nkb i=1, where nkb represents the size of knowledge base. The data in the knowledge base will use the same normalization as the input. To maintain the generalization capabilities of the foundation model, we use multi-domain datasets for training, similar to the pre-training phase of the foundation model, which will be detailed in Section IV. [URL 🔗](#page-0)

Subsequently, we apply a sliding window with the same window size as the input of the foundation model across the training datasets. Based on the scale of each sub-dataset, we ultimately establish a knowledge base in which each domain has an equal proportion to uphold the balance. Additionally, there is no overlap present in the data within the knowledge base. During Training, to prevent data leakage caused by accessing future sequences, the retriever is constrained to retrieve infor- mation solely from datasets that are different from the dataset of input sequence. During inference, TimeRAF has the option to utilize the extensive multi-domain knowledge base that we have developed or to opt for a domain-specific dataset as the knowledge base, based on the specific requirements.

## E. Knowledge Retrieval

Inspired by DPR [14], we employ a dual-encoder retriever to efficiently obtain relevant information from the knowledge base. [URL 🔗](#page-0)

1) Knowledge Retrieval Learning: The retriever adopts a MLP-based encoder to respectively embed the query and the candidates. In TimeRAF, we utilize the input directly as the query. Then, the retriever calculates the dot product similarity score between the query and each candidate using their respec- tive embeddings. Finally, the candidates with the k highest sim-

ilarity scores are retrieved, denoted as C = {c1, c2,..., ck}.

Intuitively, by augmenting the model with retrieved knowl- edge, the goal is to improve predictions based on desired metrics, such asMeanSquared Error (MSE).However, it is challenging to guarantee that retrieved candidates with higher similarity scores will consistently providemore useful knowledge for forecasting.

To address this, we employ the forecaster F as an evaluator. Since the forecaster is based on the foundation model, we can leverage its strong forecasting capability to provide feedback and guide the selection of knowledge. The remaining forecaster module (i.e. Channel Prompting) will be discussed in the sub- sequent subsection.

Specifically, using the retrieved candidate ci,weemploythe forecaster F to obtain the the metric values of prediction ˆy =

F(x, ci).If F finds that integrating the knowledge from ci is beneficial for forecasting, we encourage the retriever to rank the score of ci to be higher. In this way, the model can automatically decide the usefulness of the candidates and learn to retrieve more helpful candidates from the knowledge base. To implement this learning strategy, we first transform the metric values into a probability distribution as:

where M(ˆy, y) denotes the metric function to evaluate the quality of the prediction ˆy given the ground truth y and τm is a temperature hyperparameter to control the sensitivity of the metric. pi serves as an approximate measure of the contribution of ci to the final prediction. Here the metric function satisfies that a higher value of M(·, ·) indicates better performance. If a smaller value of M(·, ·) indicates better performance, we can [URL 🔗](#page-0)

replace M(·, ·) with −M(·, ·) in (1). [URL 🔗](#page-0)

It is evident that a beneficial ci will correspond to a higher pi, allowing pi to serve as a supervised signal to guide the learning of the retriever. In particular, we aim to align the similarity score

generated by the retriever withP = {pi}k i=1. Formally, suppose wehave top-k retrieval candidatesCq along with their associated

retrieval scoresSq ∈ Rk with respect to the queryq.We can then aim to minimize the Kullback-Leibler divergence between Sq and P as follows:

LR = DKL (P, softmax (Sq/τs)) ,

where DKL denotes the KL divergence and τs is a temperature hyperparameter to control the sensitivity ofthe similarity scores.

However, during the training process, there is a risk that the retriever may become entrenched in a local optimum, thereby consistently retrieving a limited set or a narrow range of can- didates. Consequently, the forecaster fails to learn from the retriever and disregards the retrieved knowledge. To address this issue, we employ a straightforward augmentation strategy by incorporating randomly sampled data from the knowledge base to promote a broader exploration of candidates within the framework. Specifically, we initially replace each ci with a randomly selected candidate caug i at a probability of ρ, yielding Caug q . Then the dot product similarity between the query q and each candidate caug i will be updated as the retrieval scores

Saug q = {saug i }k i=1. Finally, based on (2), we can minimize the [URL 🔗](#page-0)

following loss to update the retriever:

2) Retriever-Forecaster Joint Training: Utilizing the candi- dates retrieved by the retriever, we aim to enhance forecasting


capability by leveraging external knowledge and further super- vising the training of the forecaster. As illustrated in Fig. 2, the backbone of foundation model remains frozen throughout the training process. To maintain consistency, we employ the same prediction loss utilized during the pre-training phase to update the entire forecaster. Formally, the prediction loss can be formulated as follows: [URL 🔗](#page-0)

LPred = LPretrain (F(x,C), y) .

Combined with the loss utilized for updating the retriever, the whole training loss is

L = LPred + λ ·Laug R ,

where λ is a weight hyperparameter of Laug R .

## F. Knowledge Integration

Given k retrieved time series dataC = {c1, c2,..., ck} from the knowledge base, we propose Channel Prompting as an integration methods to leverage the valuable information within them, thereby complementing the pre-trained knowledge of TSFMs to enhance forecasting performance.

Following the preprocessing procedure of the TSFM, each sequence ci inC will undergo normalization followed by patch- ing, analogous to the input. Thereafter, these patches will be processed through a projection layer to derive their respective

embeddings. Let x ∈ Rn×d denotes the input embedding and date. Here, ci ∈ Rn×d n represents the embedding of the ith retrieved candi- denotes to the number of patches, while d indicates the dimensionality of the embedding.

The Channel Prompting begins with a flatten operation on both embedding ofinput and retrieved candidates. Subsequently, the flattened embedding of input and retrieved candidate will be concatenated:

zi = Concat (Flatten (x) , Flatten ( ci)) .

By integrating the input embedding with the external knowl- edge embedding, the representation of the input is enriched with supplementary contextual information. Furthermore, after

obtaining the concatenated embedding zi ∈ R2∗n∗d, the foun- dation model is better positioned to comprehend the lookback window through the incorporation of domain-specific knowl- edge or external facts.

Subsequently, we employ a MLP to effectively extract and combine the most relevant information from both the lookback window and the retrieved candidates. This process enables the compression of the combined representation into a more meaningful and compact form. In particular, the concatenated embedding zi is compressed back to the original dimensions

corresponding to the foundation model, yielding Besides, the original input embedding is reintroduced through zi ∈ Rn×d. a residual connection to ensure the complete preservation of the information from the lookback window. Since k retrieved

time series data yields k concatenated embeddings {zi}k i=1,we will extract the valuable feature from the combined embedding zi first. Then, to retain useful information from all k retrieved

sequences, the features extracted from these sequences are av- eraged. The entire process can be formulated as follows:

x∗ = x + z = x + Avg (MLP(z1),..., MLP(zk))

This operation ensures that the model captures information from all retrieved data while balancing their contributions. By extracting the relevant feature of the concatenated embedding, the foundation model is empowered to incorporate additional contextual information from the knowledge base. Consequently,

the final knowledge-enhanced input embedding

x∗ ∈ Rn×d

will

be fed into the foundation model backbone, thereby enhancing

prediction accuracy.

## G. Inference Procedure

During the inference process, given a query, candidates with the highest k retrieval scores from the knowledge base are retrieved by the retriever. Following preprocessing, the embed- dings of the input and the retrieved candidates are processed through Channel Prompting to effectively integrate external knowledge. Ultimately, the knowledge-enhanced embeddings are fed into the backbone of the time series foundation model, which then generates the final prediction.

## IV. EXPERIMENT

## A. Experiments Setups

Training Datasets and Knowledge Base: Our training em- ploys a subset ofabout 320 million time points fromLOTSA[12] and UTSD [13], which were used for the pre-training of Time Series Foundation Models. The dataset encompasses a diverse range of domains to maintain the generalization capabilities of the foundation model. The knowledge base used for training contains approximately 3 million data points, as introduced in Section III-D, selected from the training datasets. Each domain within the knowledge base is designed to contain a roughly equivalent number of data points to maintain balance. The com- plete list of training datasets and knowledge base is presented in Table II. To enhance data integrity, missing values are sys- tematically addressed using linear interpolation techniques. For each univariate, multivariate, or irregular-sampled time series, we store them with timestamps, domains, sampling frequencies and other meta-information in one directory using ARROW[38] format. One dataset may composed of multiple related time series. [URL 🔗](#page-0)

All datasets can be classified into six distinct domains by their source: Energy, Nature, Transport, Web, Sales, and Healthcare. The datasets exhibit diverse sampling frequencies, ranging from macro intervals such as daily to more fine-grained intervals like hourly and minutely.

Evaluation Datasets: For zero-shot evaluation, we consider the popular long sequence forecasting benchmark, including six public datasets : ETTh1, ETTh2, ETTm1, ETTm2, Weather, and Electricity, which are commonly utilized in previous works [12], [39]. It should be noted that all evaluation datasets are inaccessible during the training process. [URL 🔗](#page-0)


*TABLE II*

*DATASET DETAILED DESCRIPTIONS*

| | | Dataset |   |   |   |
| --- | --- | --- | --- | --- |
| Domain |   | Frequency | Time Points | Source |
| Energy | Australian Electricity Demand | BDG-2 Fox | H | 2,324,568 | BuildingsBench [32] |
|   |   | 307 4s | 1,153,584 | Monash [33] |
|   | Solar Power |   | 7,397,222 | Monash [33] |
|   |   |   | 7,094,304 1,129,444 |   |
|   | Los-Loop | sT |   | LibCity [34] |
| P Uber TLC | Hourly | H |   | [35] GluonTS |
| Nature | Subscasonal Precipitation Saugeen | D | 9.760426 | SubscasonalClimateUSA library [36] |
|   |   | D | 23,711 | Monash [33] |
| Daily | Kaggle Web Traffic | D | 116,485,589 40,619,100 | Monash [33] |
|   | Wiki-Rolling | D |   | GluonTS [35] |
| Sales | Ms | D | 58.327.370 | GluonTS [35] |
|   | Favorita Transactions | D | 84,408 | Kaggle |
|   | Motorlmagery |   | 72,576,000 |   |
| Healthcare |   | 0 |   | UCR Time Series Archive [37] |
|   | US Births | D | 7.275 | Monash [33] |

Time Ports denotes the total number of Gime aggregating from all variates IT

denotes the original paper or resource of the dataset.

Frequency denotes the sampling interval of Gme ports. Source

*Fig. 3. Improvement by TimeRAF on zero-shot forecasting. 5% Few shot denotes finetuning TSFM with 5% of downstream dataset. TimeRAF demonstrates significant improvements across various datasets, even outperforming results obtained by few-shot fine-tuning.*

Metric:We employ mean squared error (MSE) as the standard error metric for our experiments.

Implementation Detail: We employ TTM-Base (TTMB), one of the latest State of The Art (SOTA) TSFM, as our backbone. The input context length is set to 512 and the forecasting length is 96 in most of our experiments except for the results of multiple prediction horizons in Table IV. During inference, TimeRAF uses the same knowledge base employed during the training phase. Except for the experiments in Section IV-D2, we select 8 candidates for each input. [URL 🔗](#page-0)

Baselines: We compare with 6 of the latest open-sourced state-of-the-art foundation models: TTM [23],Moirai [12], MOMENT [40],Timer [13], Chronos [41], TimesFM [20]. ( Except for the TTM results based on our reproduction,1 the other results are sourced from the original paper of Timer [13] and Moirai [12]. [URL 🔗](#page-0)

## B. Results ofZero-Shot Forecasting

ImprovementbyTimeRAFon zero-shot forecasting: As shown in Fig. 3, we demonstrate the improvements brought by our [URL 🔗](#page-0)

method in zero-shot forecasting. The yellow bar represents the scenario where 5% of the training data from the dataset is used to fine-tune the foundation model backbone. Augmented by retrieved knowledge, our TimeRAF presents significant im- provements across all the datasets. The experiment results in- dicate that, through our training, our retriever has learned to search for valuable information from the knowledge base. Sub- sequently, through channel prompting, TimeRAF successfully extracts useful knowledge, ultimately enhancing the prediction results. Moreover, TimeRAF also outperforms the performance achieved through few-shot fine-tuning, which further demon- strate the effectiveness of our method.

TimeRAF vs. other models:We compare TimeRAF against 11 baselinemodels. The experimental results are shown in Table III. Compared to the foundation models, TimeRAF achieves either the best or competitive results across multiple datasets. Besides, our method is an enhancement built upon the foundation model. As the foundation model continues to evolve, TimeRAF is anticipated to yield further improvements when adapted to new backbones. [URL 🔗](#page-0)

TimeRAFwith multiple prediction horizons: The full results of multiple prediction horizons are presented in Table IV. Results are averaged over 5 runs, with mean and std reported. From [URL 🔗](#page-0)


*TABLE III*

*FULL RESULTS OF ZERO-SHOT FORECASTING EXPERIMENTS*

*TABLE IV*

*FULL FORECASTING RESULTS:TIMERAF DELIVERS IMPROVEMENTS ACROSS*

*ALL FORECAST HORIZONS*

these results, we observe that TimeRAF consistently delivers significant improvements over the backbone across all forecast horizons, further demonstrating the method’s robustness and generality.

## C. Ablation Studies

1) Effectiveness of the Retriever: As described in Section III-E, we employ an end-to-end approach to train the retriever, encouraging it to select the most valuable candidates from the knowledge base. To validate the effectiveness of the learnable retriever, we have designed two baselines for compar- ison: one that randomly selects candidates from the knowledge base and another that selects the top k candidates based on cosine [URL 🔗](#page-0)

*TABLE V*

*ABLATION STUDIES ON RETRIEVER AND CHANNEL PROMPTING*

similarity. As shown in Table V, randomly selecting candidates fails to provide useful information to the forecaster andmay even introduce noise, degrading the model’s predictive performance. While the cosine similarity-based retrieval method offers some knowledge, its improvement is limited and falls short compared to our method, which automatically learns how to retrieve useful knowledge. [URL 🔗](#page-0)

2) Channel Prompting: An effective integration method, named Channel Prompting, is used to extract the relevant knowl- edge from the retrieved data, as detailed in Section III-F.Toval- idate the effectiveness of channel prompting, we establish three baselines for comparison: the first, called Token-Concat, entails concatenating the retrieved candidates with the input at the token level, while the second, termed Average, involves directly computing the mean of the candidate and input embeddings for integration. The third one uses input embedding as query and retrieved candidate embedding as key and value to compute the cross attention. As shown in TableV, our TimeRAF outperforms both baselines. The token-level concatenation imposes restric- tions on the integration to tokens located in the same position. While averaging input and retrieved candidates embeddings prove insufficient for extracting valuable information. [URL 🔗](#page-0)

To compare different weighting strategies for embeddings in Channel Prompting, we evaluated the results of using a uniform weighting strategy (TimeRAF) against those of score-based and distance-based weighting strategies, as shown in Table VI. The score-based approach assigns weights by applying softmax to the retrieval scores between the query and retrieved data embeddings, where higher scores lead to larger weights. Similarly, the distance-based approach uses softmax on the [URL 🔗](#page-0)


*TABLE VI*

*ABLATION STUDY ON WEIGHTING MECHANISMS IN CHANNEL PROMPTING*

*BEST RESULTS ARE HIGHLIGHTED IN BOLD*

*TABLE VII*

*ABLATION STUDY ON CANDIDATE AUGMENTATION*

*TABLE VIII*

*ALTERNATIVE BACKBONE RESULTS:USING TIMER1B [13] AS THE BACKBONE, OUR RAF CONSISTENTLY DELIVERS IMPROVEMENTS OVER THE BACKBONE [URL 🔗](#page-0)*

*ACROSS ALL DATASETS*

embedding distances, with smaller distances resulting in larger weights. From the results, it is evident that our uniform weight- ing strategy, despite its simplicity, proves to be a more effective approach. As illustrated in Fig. 6(b), retrieved samples with only moderate similarity can still provide crucial predictive signals. Uniform weighting helps preserve the contribution of such informative samples, which may be undervalued by similarity weighting. [URL 🔗](#page-0)

3) Candidate Augmentation: During the training process, there is a risk that the retriever may become entrenched in a local optimum, thereby consistently retrieving a limited set or a narrow range of candidates. To address this issue, we employ a straightforward augmentation strategy. We provide experi- ments results of TimeRAF without candidate augmentation in Table VII. [URL 🔗](#page-0)

4) Alternative Backbone: To further evaluate the effective- ness of RAF, we conducted additional experiments using Timer1B [13] as the backbone. The results are provided in the Table VIII. These experiments demonstrate that our method consistently delivers significant improvements across different backbone models. [URL 🔗](#page-0)

## D. Model Analysis

1) Choice of Knowledge Base: Source of Knowledge Base: The previous experimental results have convincingly demonstrated that following training, TimeRAF has acquired

*TABLE IX*

*COMPARISON OF TIMERAF WITH VARIOUS KNOWLEDGE BASES*

*Fig. 4. Impact of knowledge base size. Smaller knowledge base provides less useful information, worsening results.*

the capability to dynamically access pertinent knowledge from a external knowledge base and effectively leverage this valuable information. To further explore the implications of employing various knowledge bases during inference, we have devised the following three scenarios: (a)TimeRAFR randomly selects data from the pre-trained multi-domain dataset, which may result in an uneven distribution across different domains. (b)TimeRAFD utilizes a knowledge base closely related to the test data. Specifi- cally, the training set from the same dataset is directly employed as the knowledge base for retrieval. (c) TimeRAF engages a meticulously curated multi-domain dataset, which is detailed in Table II. As presented in Table IX, TimeRAF achieves the best performance across different datasets. As a specifically designed knowledge base, it encompasses a rich repository ofin- formation across multiple domains, enabling it to provide useful information to enhance predictions. Meanwhile, the knowledge base used in TimeRAFD is particularly relevant to the test data, providing domain-specific knowledge. As a result, the zero-shot time series forecasting performance achieved with this knowledge base ranks just belowthat ofTimeRAF. However, the randomly selected knowledge base used in TimeRAFR suffers from domain imbalance, which limits the enhancement that external knowledge can provide to the forecaster. [URL 🔗](#page-0)

Size of Knowledge Base: The size of knowledge base also plays a vital role in the framework, determining the extent of external knowledge that can be accessed. We perform a comprehensive analysis ofthis aspect, presenting average results across various datasets in Fig. 4. Initially, both TimeRAF and [URL 🔗](#page-0)


*Fig. 5. Influence of the Candidates Number k. As k increases, the performance gradually improves due to the integration ofmore relevant knowledge. However, when k exceeds a certain threshold, the abundance of information can introduce redundancy, negatively affecting the prediction.*

*TABLE X*

*CROSS DOMAIN KNOWLEDGE BASE*

TimeRAFD utilize knowledge bases of comparable scale, each consisting of approximately 3 million data points, as outlined in Section IV. Then, we progressively reduce the size of the knowledge base. As shown in Fig. 4, the MSE are influenced by modifications in the knowledge base size. As the size diminishes, the amount ofexternal knowledge it can provide decreases, lead- ing to a decline in the performance. Once the knowledge base is reduced beyond a certain point, using a domain-specific knowl- edge base (TimeRAFD) can provide more relevant information compared to a multi-domain knowledge base (TimeRAF), re- sulting in better forecasting performance. The findings under- score the importance of selecting the appropriate knowledge base size tailored to the specific application at hand. [URL 🔗](#page-0)

Cross Domain Knowledge Base: We conduct a supplemen- tary experiment in which we used a knowledge base from a domain different from the test data domain. The results are provided in Table X. Specifically, we evaluated the model on the ETTh1, ETTh2, ETTm1, ETTm2 (Energy domain) using knowledge base from another datasetWeather (Nature domain). Here, “Domain Specific KB” refers to using the training set of the dataset to construct the knowledge base. The results above show that with a knowledge base from a distinct domain, TimeRAF still outperforms the baseline where no knowledge base is used. Additionally, retrieving from the Domain Specific Knowledge Base achieves the better performance. These find- ings indicate that using a knowledge base containing data from the same domain as the input yields better results. The results also underscore the importance of knowledge base selection and demonstrate our method’s ability to effectively identify and utilize valuable information from diverse knowledge bases, even in novel domain scenarios. [URL 🔗](#page-0)

2) Influence of the Candidates Number: We investigate the impact of varying the numbers of retrieved candidates on pre- diction performance. As illustrated in Fig. 5, using multiple retrieved candidates (e.g., 4 or 8) equips the forecaster with [URL 🔗](#page-0)

*TABLE XI*

*MODEL EFFICIENCY COMPARISON:WE PROVIDE MODEL SIZE AND PER BATCH*

*CPU INFERENCE TIME OF EACH FOUNDATION MODEL*

a more comprehensive set of external information compared to relying on a single candidate, thereby further enhancing prediction performance. Nevertheless, the performance gains do not persist as the variable k increases. In our analysis of the test data, we observe that when k is elevated to 16 or 32, there is no significant improvement in the model’s prediction accuracy. This phenomenon may be attributable to the introduction of excessive candidates, which can lead to redundant information, ultimately detracting from the overall effectiveness of the pre- diction results. This saturation point suggests that the benefits of diversity among the retrieved candidates can be counteracted by the presence of overlapping or irrelevant data. When too many candidates are presented, the model may struggle to discern the most salient features contributing to accurate predictions. The noise introduced through excessive candidates can obfuscate valuable insights, resulting in diminished performance.

## E. Model Efficiency Comparison

The comparison of model efficiency is presented in Table XI. The supplementary modules incorporated in RAF are designed to be lightweight, contributing only a minimal increase in pa- rameters. Furthermore, the retrieval process is based on a dot product calculation, which enhances efficiency. Consequently, TimeRAF maintains a satisfactory model size and inference time. Additionally, the modular architecture ofTimeRAF allows for seamless integration of new features without compromising overall performance. This adaptability is crucial, as it enables rapid experimentation and fine-tuning, ensuring that the model can evolve in response to emerging challenges and datasets. The lightweight design not only facilitates quicker training cycles but also supports deployment in resource-constrained environments, making TimeRAF a versatile choice for various applications. [URL 🔗](#page-0)

## F. Case Study on Retrieved Knowledge

To conduct a detailed analysis of the information provided by the retriever and its contribution to enhancing the zero-shot


*Fig. 6. Case Study on Retrieved Knowledge. (a) Example A: The retrieved knowledge shares similar periodicity and subtle fluctuations with the input, facilitating the forecaster’s ability to effectively capture the prior knowledge inherent in the input, thereby improving prediction performance. (b) Example B: The retrieved data provides supplementary insights, including partial future information (highlighted within the red dashed box), empowering the forecaster to generate better predictions.*

*Fig. 7. Visualization ofzero-shot forecasting across different datasets. The visualization includes all the evaluation datasets: ETTh1, ETTh2, ETTm1, ETTm2, Weather, and Electricity.*

forecasting capabilities of the foundation model, we present two illustrative examples in Fig. 6. As shown in Fig. 6(a),the retrieved knowledge exhibits similar periodicity and nuanced fluctuations to the input, enhancing the forecaster’s capacity to effectively capture the prior knowledge inherent in the input data, thereby improving prediction performance. [URL 🔗](#page-0)

The data retrieved by the retriever is not always highly similar to the input, as illustrated in Fig. 6(b). In the absence of the retrieval-augmented forecasting method, the model generates predictions with small amplitude, relying solely on constrained historical data and underlying inertia. However, the incorpo- ration of retrieved data provides additional insights, including [URL 🔗](#page-0)

partial future information (highlighted within the red dashed box), thereby improving the prediction.

## G. Forecasting Visualization

We provide several visualization of zero-shot forecasting in Fig. 7. These visualizations illustrate the effectiveness of our proposed method based on leveraging external knowledge. Each subplot in Fig. 7 captures distinct scenarios, allowing for a comprehensive understanding of the model’s capabilities under different conditions. [URL 🔗](#page-0)


## V. CONCLUSION AND FUTURE WORK

In this paper, we introduce TimeRAF, a novel frame- work designed to leverage retrieval-augmented generation for zero-shot time series forecasting. We develop customized time series knowledge bases and employ an end-to-end learnable retriever to extract valuable information. We also introduce Channel Prompting for knowledge integration. By leveraging external knowledge, TimeRAF exhibits a notable enhancement in zero-shot time series forecasting.

While TimeRAF achieves phenomenal performance, this rep- resents merely the initial step in the integration of time series methods and RAG. Due to resource constraints, the knowledge base is established based on original time series data without the implementation of advanced techniques like trend-seasonal decomposition. In terms of architecture, our approach to inte- grate external knowledge is somewhat heuristic and future work should design a more flexible and elegant approach. Also, the current architecture has ignored the potential interdependencies among different channels, which could be addressed more ef- fectively in future methods.

- [14] V. Karpukhin et al., “Dense passage retrieval for open-domain question answering,” 2020, arXiv: 2004.04906.

- [15] W. Chen, H. Hu, C. Saharia, and W. W. Cohen, “Re-imagen: Retrieval- augmented text-to-image generator,” in Proc. 11th Int. Conf. Learn. Rep- resentations, 2023.

- [16] Q. Huang et al., “Learning retrieval augmentation for personalized dia- logue generation,” in Process., 2023, pp. 2523–2540. Proc. 2023 Conf. Empirical Methods Natural Lang.

- [17] A. Garza and M. Mergenthaler-Canseco, “TimeGPT-1,” 2023, arXiv: 2310.03589.

- [18] K. Rasul et al., “Lag-Llama: Towards foundation models for time series forecasting,” 2023, arXiv:2310.08278.

- [19] H. Touvron et al., “Llama: Open and efficient foundation language mod- els,” 2023, arXiv:2302.13971.

- [20] A. Das, W. Kong, R. Sen, and Y. Zhou, “A decoder-only foundation model for time-series forecasting,” in Proc. 41st Int. Conf. Mach. Learn., 2024.

- [21] Y. Liu, G. Qin, X. Huang, J.Wang, andM. Long, “Timer-XL: Long-context transformers for unified time series forecasting,” in Proc. Int. Conf. Learn. Representations, 2025.

- [22] X. Liu et al., “Unitime: A language-empowered unified model for cross-domain time series forecasting,” in Proc. ACM Web Conf., 2024, pp. 4095–4106.

- [23] V. Ekambaram et al., “Tiny time mixers (TTMs): Fast pre-trained mod- els for enhanced zero/few-shot forecasting of multivariate time series,” 2024, arXiv:2401.03955.

- [24] S. Zhou, U. Alon,F. F. Xu,Z. JIang, andG.Neubig, “Doccoder: Generating code by retrieving and reading docs,” 2022, arXiv:2207.05987.

- [25] W. Shi et al., “REPLUG: Retrieval-augmented black-box language mod- els,” 2023, arXiv:2301.12652.

## ACKNOWLEDGMENT

This work was done during internship at Microsoft Research.

- [26] H. Liu et al., “Learning customized visual models with retrieval- augmented knowledge,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 15148–15158.

- [27] T. Wang and G. Cui, “RATSF: Empowering customer service vol- ume management through retrieval-augmented time-series forecasting,” 2024, arXiv:2403.04180.

## REFERENCES

- [1] Y. Zhang et al., “OneNet: Enhancing time series forecasting models under concept drift by online ensembling,” in Proc. 37th Conf. Neural Inf. Process. Syst., 2023, pp. 69949–69980.

- [2] H. Zhang, Y.-F. Zhang, Z. Zhang, Q.Wen, and L. Wang, “LogoRA: Local- global representation alignment for robust time series classification,” IEEE Trans. Knowl. Data Eng., vol. 36, no. 12, pp. 8718–8729, Dec. 2024. [Online]. Available: https://doi.ieeecomputersociety.org/10.1109/TKDE. 2024.3459908 [URL 🔗](https://doi.ieeecomputersociety.org/10.1109/TKDE.2024.3459908)

- [3] H. Li et al., “Bridge: Bootstrapping text to control time-series gener- ation via multi-agent iterative optimization and diffusion modelling,” 2025, arXiv:2503.02445.

- [4] Z. Chen, D. Chen, X. Zhang, Z. Yuan, and X. Cheng, “Learning graph structures with transformer for multivariate time-series anomaly detection in IoT,” IEEE Internet Things J., vol. 9, no. 12, pp. 9179–9189, 2022.

- [5] X. Yu, Z. Chen, Y. Ling, S. Dong, Z. Liu, and Y. Lu, “Tempo- ral data meets LLM–explainable financial time series forecasting,” 2023, arXiv:2306.11025.

- [6] J. Li, C. Liu, S. Cheng, R. Arcucci, and S. Hong, “Frozen language model helps ECG zero-shot learning,” in Proc. Med. Imag. Deep Learn., 2024, pp. 402–415.

- [7] H.Wu, H. Zhou, M. Long, and J.Wang, “Interpretable weather forecasting for worldwide stations with a unified deep model,” Nature Mach. Intell., vol. 5, no. 6, pp. 602–611, 2023.

- [8] K. Jin, J. Wi, E. Lee, S. Kang, S. Kim, and Y. Kim, “TrafficBERT: Pre- trained modelwith large-scale data for long-range traffic flowforecasting,” Expert Syst. Appl., vol. 186, 2021, Art. no. 115738.

- [9] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, “A time series is worth 64 words: Long-term forecasting with transformers,” in Proc. 11th Int. Conf. Learn. Representations, 2023.

- [10] A. Zeng, M. Chen, L. Zhang, and Q. Xu, “Are transformers effective for time series forecasting?,” in Proc. AAAI Conf. Artif. Intell., 2023, pp. 11121–11128.

- [11] Y. Liang et al., “Foundation models for time series analysis: A tutorial and survey,” in Proc. 30th ACMSIGKDD Conf. Knowl. Discov. Data Mining, 2024, pp. 6555–6565.

- [12] G. Woo, C. Liu, A. Kumar, C. Xiong, S. Savarese, and D. Sahoo, “Unified training of universal time series forecasting transformers,” 2024, arXiv:2402.02592.

- [13] Y. Liu, H. Zhang, C. Li, X. Huang, J. Wang, and M. Long, “Timer: Generative pre-trained transformers are large time series models,” in Proc. 41st Int. Conf. Mach. Learn., 2024, pp. 32369–32399.

- [28] B. Jing et al., “Retrieval based time series forecasting,” 2022, arXiv: 2209.13525.

- [29] K. Tire, E. O. Taga, M. E. Ildiz, and S. Oymak, “Retrieval augmented time series forecasting,” 2024, arXiv:2411.08249.

- [30] Y. Ang, Y. Bao, Q. Huang, A. K. Tung, and Z. Huang, “TSGAssist: An interactive assistant harnessing LLMs and rag for time series generation recommendations andbenchmarking,” inProc.VLDBEndowment, vol. 17, no. 12, pp. 4309–4312, 2024.

- [31] J. Liu, L.Yang, H. Li, and S. Hong, “Retrieval-augmented diffusionmodels for time series forecasting,” in Proc. Adv. Neural Inf. Process. Syst., 2024, pp. 2766–2786.

- [32] P. Emami, A. Sahu, and P. Graf, “BuildingsBench: A large- scale dataset of 900 k buildings and benchmark for short-term load forecasting,” in Proc. Adv. Neural Inf. Process. Syst., 2023, pp. 19823–19857.

- [33] R. Godahewa, C. Bergmeir, G. I. Webb, R. J. Hyndman, and P. Montero- Manso, “Monash time series forecasting archive,” 2021, arXiv:2105. 06643.

- [34] J. Jiang, C. Han, W. Jiang, W. X. Zhao, and J. Wang, “Towards efficient and comprehensive urban spatial-temporal prediction: A unified library and performance benchmark,” 2023, arXiv:2304.14343.

- [35] A. Alexandrov et al., “GluonTS: Probabilistic and neural time series modeling in python,” J.Mach.Learn.Res., vol. 21, no. 116, pp. 1–6, 2020.

- [36] S. Mouatadid et al., “Subseasonalclimateusa: A dataset for subseasonal forecasting and benchmarking,” in Proc. Adv. Neural Inf. Process. Syst., 2024, pp. 7960–7992.

- [37] H. A. Dau et al., “The UCR time series archive,” IEEE/CAA J. Automatica Sinica, vol. 6, no. 6, pp. 1293–1305, Nov. 2019.

- [38] N. Richardson et al., arrow: Integration to ‘Apache’ ‘Arrow,’ r package ver- sion 17.0.0, 2024, https://arrow.apache.org/docs/r/. [Online]. Available: https://github.com/apache/arrow/

- [39] M. Jin et al., “Time-LLM: Time series forecasting by reprogramming large language models,” in Proc. 12th Int. Conf. Learn. Representations, 2023.

- [40] M. Goswami, K. Szafer, A. Choudhry, Y. Cai, S. Li, and A. Dubrawski, “Moment: A family of open time-series foundation models,” 2024, arXiv:2402.03885.

- [41] A. F. Ansari et al., “Chronos: Learning the language of time series,” 2024, arXiv:2403.07815.


Huanyu Zhang is currently working toward the PhD degree in computer science with the New Laboratory of Pattern Recognition (NLPR), State Key Labora- tory of Multimodal Artificial Intelligence Systems (MAIS), Institute of Automation, Chinese Academy of Sciences (CASIA). His current research interests mainly include time series analysis.

Chang Xu received the bachelor’s degree from Nankai University, in 2014, and the PhD degree through the joint program ofMicrosoft Research Asia and Nankai University, in 2019. She is a senior re- searcher with Microsoft Research, focusing on AI in Finance, time series analysis, and generative model- ing. Her academic contributions include publications in refereed journals and conferences such as IEEE Transactions on Computers, ICLR,KDD,ACMMM, WWW, AAAI, IJCAI, ICME, and CIKM.

Yi-Fan Zhang is currently working toward the PhD degree in computer science with the New Laboratory of Pattern Recognition (NLPR), State Key Labora- tory of Multimodal Artificial Intelligence Systems (MAIS), Institute of Automation, Chinese Academy of Sciences (CASIA). His current research interests mainly include robust and reliable machine learning (ML) systems.

Zhang Zhang (Member, IEEE) received the PhD de- gree from the National Laboratory of Pattern Recog- nition (NLPR), Institute of Automation, Chinese Academy of Sciences (CASIA), in 2009. From 2009 to 2010, he was a research fellow with the School of Computer Science and Engineering, Nanyang Tech- nological University (NTU). In September 2010, he joined the NLPR, Institute of Automation, Chinese Academy of Sciences(CASIA). Now, he is an asso- ciate professor with the New Laboratory of Pattern Recognition (NLPR), CASIA.

Liang Wang (Fellow, IEEE) received both the BEng and MEng degrees from Anhui University, in 1997 and 2000, respectively, and the PhD degree from the Institute of Automation, Chinese Academy of Sci- ences (CASIA), in 2004. From 2004 to 2010, he was a research assistant with Imperial College London, United Kingdom, and Monash University, Australia, a research fellow with the University of Melbourne, Australia, and a Lecturer with the University of Bath, United Kingdom, respectively. Currently, he is a full professor of the Hundred Talents Program with the

State Key Laboratory of Multimodal Artificial Intelligence Systems, CASIA. He has widely published in highly ranked international journals such as IEEE Transactions on Pattern Analysis and Machine Intelligence and IEEE Transac- tions on Image Processing, and leading international conferences such as CVPR, ICCV, and ECCV. He has served as an associate editor of IEEE Transactions on Pattern Analysis and Machine Intelligence, IEEE Transactions on Image Processing, and PR. He is an IAPR Fellow.

Jiang Bian received the bachelor’s degree from Peking University, China, and the PhDdegree in com- puter science from the Georgia Institute of Technol- ogy, USA. He is a senior principal research manager with Microsoft Research. He is leading the machine learning solutions and services group, with the main focus on designing cutting-edge machine learning algorithms into real-world application scenarios, in- cluding finance, healthcare, supply-chain and sustain- ability. Prior to this, he was a scientist with Yahoo! Labs in the United States, responsible for the content

optimization and personalization andWeb search modules inYahoo! Homepage. After that, he jointed a leading content distribution platform in China, i.e., Yidian Inc., and became one of the core members of this startup company, with the major responsibility of developing advanced recommendation models. He has authored tens of research papers in many well-recognized international conferences and has submitted a couple of US patents. He has also served as PC member for several international conferences and Peer Reviewer for a few well-known journals.
