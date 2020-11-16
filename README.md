# Writing by Memorizing: Hierarchical Retrieval-based Medical Report Generation

Code base for paper "Writing by Memorizing: Hierarchical Retrieval-based Medical Report Generation".
We propose **MedWriter** for report generation

## File Structure
    data/           : data split, reports, vocabulaies and
                                           classification labels for two datasets
        open-i/
        mimic/
    model/          : model files for LLR module, VLR module and full MedWriter
        LLR.py
        VLR.py
        medicalwriter.py
        mlclassifier.py
    utils/          : dataloaders and tool functions
        dataloader.py
        tools.py
    my_build_vocab.py
    evaluate.py
    train_classifier.py
    train_LLR.py
    train_VLR.py
    train_MedWriter.py

## Usage

1. Pretrain Visual-Language Retreival (VLR) Module

        python train_VLR.py

2. Pretrain Language-Language Retreival (LLR) Module

        python train_LLR.py

3. Train the full model

        python train_MedWriter.py
<!--
**MedWriter/MedWriter** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
