# Participation Role-Driven Engagement Estimation of ASD Individuals in Neurodiverse Group Discussions
<div>
  <img src="assets/model.png">
  <p></p>
</div>

This repository contains the official PyTorch implementation for the paper 
[Participation Role-Driven Engagement Estimation of ASD Individuals in Neurodiverse Group Discussions](https://) (ICMI 2024).

## References
If this work is useful for your research, please consider citing it.
```bibtex
@inproceedings{10.1145/3678957.3685721,
author = {Stefanov, Kalin and Nakano, Yukiko I. and Kobayashi, Chisa and Hoshina, Ibuki and Sakato, Tatsuya and Nihei, Fumio and Takayama, Chihiro and Ishii, Ryo and Tsujii, Masatsugu},
title = {Participation Role-Driven Engagement Estimation of ASD Individuals in Neurodiverse Group Discussions},
year = {2024},
isbn = {9798400704628},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3678957.3685721},
doi = {10.1145/3678957.3685721},
abstract = {Adults with autism spectrum disorder (ASD) face difficulties in communicating with neurotypical people in their daily lives and workplaces. In addition, research on modeling communication in neurodiverse groups is scarce. To recognize communication difficulties caused by neurodiversity, we first, collected a multimodal corpus for decision-making discussions in neurodiverse groups that included a person with ASD and two neurotypical participants. For corpus analysis, we investigated eye-gaze and facial expression exchanges between individuals with ASD and neurotypical participants during both listening and speaking. The findings were extended to automatically estimate the engagement of ASD individuals. To capture the effect of contingent behaviors between ASD individuals and neurotypical participants, we developed a transformer-based model that considers the participation role by changing the direction of cross-person attention depending on whether the ASD individual is listening or speaking. The proposed approach yields comparable results to the state-of-the-art for engagement estimation in neurotypical group conversations while accounting for the dynamic nature of behavior influence in face-to-face interactions. The code associated with this study is available at https://github.com/IUI-Lab/switch-attention.},
booktitle = {Proceedings of the 26th International Conference on Multimodal Interaction},
pages = {556–564},
numpages = {9},
keywords = {Adults with autism spectrum disorder, engagement estimation, neurodiverse group communication, participation role},
location = {San Jose, Costa Rica},
series = {ICMI '24}
}
```

## License
This project is under the MIT license. See [LICENSE](LICENSE) for details.

## Acknowledgements
The code is based on [dondongwon/Multipar-T](https://github.com/dondongwon/Multipar-T).
