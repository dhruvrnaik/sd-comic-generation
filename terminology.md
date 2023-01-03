### This file contains abbreviations and their meanings

Lots of experiments are run, with some wack terminology in order to keep words small. This explans what they mean

```disappearing``` \
We observed that when weights were scaled by a large factor, the images started to disappear. This directory is for experimetns that try to reduce this.

```trn_normed``` \
These experiments use weights that are trained with normalization. In this case, normalization refers to subtracting the bias factor from the current latent features within the u-net.

```lim_trn_normed``` \
This experiment refers to limiting the normalization effect durring experiments. When there is a large norm effect, we observe that the images are blury. 
This is because the features are averaged, and some of the info is missing. As such, we want to divide the norm effect by the number of images (ie 8) as in the following example: 
`sample*(1-bias_factor/8) + mean_biasSamples*bias_factor`
