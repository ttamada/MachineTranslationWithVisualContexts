# Statistical and Neural Machine Translation with Visual Contexts (Bachelor’s Thesis at LMU Munich)


### Abstract
In this paper, we consider possibilities to integrate translation-related visual information into statistical and neural machine translation models. We propose a simple, novel architecture to rescore and rerank target words in SMT and NMT decoders depending on the speaker’s simple visual representation using a feedforward neural network. Our model showed the possibility of a proper reranking and translation with a context-suitable expression on a very small, evenly distributed data set. Visual contexts, translation-related information separate from the texts are crucial to handle dialogs such as movies, since the spoken texts are short and contain less information for a translation. Also, the expression quality of the translation plays an important roll for spoken languages. For instance, German, Spanish and other European languages have different pronouns and its corresponding verb conjugations depending on the politeness of the dialog. In Japanese, there are even more forms of pronouns and other expressions depending on the speaker’s gender, age, politeness, and character. In order for translation system outputs to sound more natural instead of mechanical, this paper is intended to be an initial step for this challenge.

### Visual Context Weight (Section 3.4.2) 
In the visual_context_weight directory, run the following command:
```
python decode_visual.py
```

### NMT with Visual Context (Section 3.4.3) 
1. Install tensorflow 

2. In the visual_integrated_translation directory, run the following command to start training:
```
python translate_visual.py 
```

3. After the training, run the following command to start decoding:
```
python translate_visual.py --decode
```