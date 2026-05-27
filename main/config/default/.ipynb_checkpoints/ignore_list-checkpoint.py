''' List labelled samples to ignore '''

# Consider ignoring a sample if there are False Negatives in the Ground Truth in proximity* to a 
# True Positive. The consequence of leaving these samples means that the synthetic images will 
# sample foreground fluorescence in the real images into the background of the synthetic data. 
# This could impact results - consider this a preprocessing that will improve results 
# (Sarsembayeva et al. 2025; https://doi.org/10.3390/jimaging11020050)
# 
# 
# *Consult the max sampling distance

labels_to_ignore = []
