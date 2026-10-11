A)Summary

This code repository accompanies the paper entitled "Input-Difference Search and Optimization for Polytopic Differential Neural Distinguishers on SPECK32/64". It contains the main experimental programs, trained model weights, and experiment logs. The repository aims to improve the transparency of the research process and provide resources for readers to examine and reproduce the reported results.

This study investigates automated input-difference tuple search and optimization for polytopic differential neural distinguishers (PDNDs) targeting SPECK32/64. The overall framework consists of low-Hamming-weight seed-difference screening, Proximal Policy Optimization (PPO)-based joint search of dual-polytopic difference tuples, hierarchical candidate refinement and ranking, enhanced PDND training, and last-round subkey recovery experiments.

The main programs in this repository are described below.

1. guide_search.py (full and simplified versions)
   Performs preliminary searches and screening of seed differences, providing candidate differences for subsequent joint difference-tuple search.

2. DRL_search_plus_improve.py (full and simplified versions)
   Uses PPO to jointly search for and optimize two three-difference tuples. It combines large-sample statistical refinement, lightweight neural-network evaluation, and candidate ranking to select polytopic difference tuples with promising distinguishing capability.

3. train_distinguisher.py
   Loads the selected input-difference tuples for each target round, generates training, validation, and test data, constructs and trains the enhanced PDND, evaluates distinguishing accuracy and related metrics, and saves model weights and experimental results.

4. speck32_pdnd_key_recovery_combined.py
   Loads trained PDND models to conduct last-round subkey experiments, including:
   (1) Correct-key and wrong-key response test: The same ciphertext samples are partially decrypted by one round using the correct subkey and a randomly selected incorrect subkey, respectively. The PDND classification accuracies under the two conditions are then compared.
   (2) Exhaustive last-round subkey ranking experiment: Using a 7-round PDND to analyze 8-round SPECK32/64 as the main experimental setting, all possible last-round subkey candidates are enumerated. The mean log-likelihood score is computed from neural-network outputs for each candidate, and the rank of the true subkey is recorded.

The "Log file" folder contains logs from the main experiments conducted during the research. These logs allow readers to inspect experimental parameters, execution processes, and outputs, and help verify the results reported in the paper. Independent reproduction should also take into account the actual source code, model weights, and runtime environment.

B) Environment Setup

The repository provides a Conda environment configuration file named enir.yml, which records the Python environment and software dependencies required for the experiments. After installing Conda or Miniconda, run the following command in the repository's root directory:

conda env create -f enir.yml

Then activate the environment using the name specified in the YAML file. Before running the programs, check that Python, PyTorch, NumPy, CUDA, and other components are compatible with the computing platform. Neural-network training and last-round subkey search involve substantial computation, so a suitable GPU is recommended. Actual execution time may vary depending on the hardware and software environment.

C) Experimental Reproduction and Model Weights

The recommended order for reproducing the experiments is as follows:

Run guide_search.py to obtain candidate seed differences for the target round.
Run DRL_search_plus_improve.py to jointly search, refine, and rank dual-polytopic difference tuples, and save the selected tuples.
Run train_distinguisher.py to load the corresponding tuples, train or evaluate the enhanced PDND, and save model weights and evaluation results.
Run speck32_pdnd_key_recovery_combined.py to load the trained weights and select either the correct-key/wrong-key response test or the exhaustive last-round subkey ranking mode.

In the experiments reported in this paper, the simplified search programs were used to select the optimal input-difference tuples for 5 and 6 rounds, whereas the full versions were used for 7 and 8 rounds.

The repository contains all enhanced PDND model weights saved from the reported experiments. Readers can load these weights directly to perform inference and key-recovery tests. When using the weights, ensure that the network architecture, input feature channels, target rounds, and difference-tuple configurations match those used during training.

D) Declaration and Feedback

The development and optimization of the research code were assisted by artificial intelligence large language models, including but not limited to models in the GPT family. These tools were used only as development aids. The authors were responsible for the research design, experimental configurations, result analysis, manuscript preparation, and other research activities. The authors take responsibility for the publicly released code and the research reported in the paper.

During the research, the authors repeatedly modified and debugged the code based on experimental feedback and made efforts to provide necessary comments. Despite these efforts, the released code may still contain undetected errors, compatibility issues, or incomplete comments. Readers who identify any issues are welcome to contact the authors.

E) Experimental Reproducibility Notes

The source code, model weights, and execution logs are provided to support examination and reproduction of the experimental results. However, identical numerical results cannot be guaranteed across all computing environments or random settings. Readers should compare their results with the experimental settings described in the paper and the corresponding files in the repository.
