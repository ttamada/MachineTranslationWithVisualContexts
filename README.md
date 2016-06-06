=== Visual Context Weight (Section 3.4.2) ===
In the visual_context_weight directory, type as follows:
> python decode_visual.py


=== NMT with Visual Context (Section 3.4.3) ===
First, in the visual_integrated_translation directory, type as follows to start training:
> python translate_visual.py 

After the training, type as follows to start decoding:
> python translate_visual.py --decode