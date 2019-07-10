# SocialBehavior
Social Bebehavior Modeling of 2-mice data



- [SocialBehaviorptc](https://github.com/cunningham-lab/SocialBehaviors/tree/master/SocialBehaviorptc)

  PyTorch implementation of state-space models, build on the [ssm package](https://github.com/slinderman/ssm).

  

  Currently, we have

  - ARHMM with Normal emission [notebook demo](https://github.com/cunningham-lab/SocialBehaviors/blob/master/SocialBehaviorptc/socialbehavior/examples/ARHMM_demo.ipynb)

  - ARHMM with Sigmoid-Normal emission [notebook demo](https://github.com/cunningham-lab/SocialBehaviors/blob/master/SocialBehaviorptc/socialbehavior/examples/ARSigmoidNormalHMM.ipynb)

    - Sigmoid-Noral distribution:

      - parameters: $\mu, \sigma$, lower & upper bounds $l,u$.

      - generative process:

        $z \sim N(0,1)​$

        $x = (u-l) * \sigma (\mu + \sigma z) + l ​$

      - In training the real data, another parameter $\alpha$ can be added: $x =  (u-l) * \sigma (\alpha(\mu + \sigma z)) + l$ to tune the initialization.

      - See how the parameters affect the distribution here [SigmoidNormal distribution demo](https://github.com/cunningham-lab/SocialBehaviors/blob/master/SocialBehaviorptc/socialbehavior/examples/SigmoidNormal_dist_demo.ipynb)



- check how the models work on mice data [project notebooks](https://github.com/cunningham-lab/SocialBehaviors/tree/master/SocialBehaviorptc/project_notebooks)

  - ARHMM with Gaussian observation [notebook1](https://github.com/cunningham-lab/SocialBehaviors/blob/master/SocialBehaviorptc/project_notebooks/ARHMM.ipynb), [2](https://github.com/cunningham-lab/SocialBehaviors/blob/master/SocialBehaviorptc/project_notebooks/ARHMM_lrs.ipynb) (different learing rates)

  - ARHMM with Sigmoid-Normal observation [notebook1](https://github.com/cunningham-lab/SocialBehaviors/blob/master/SocialBehaviorptc/project_notebooks/ARSigmoidNormalHMM.ipynb), [2](https://github.com/cunningham-lab/SocialBehaviors/blob/master/SocialBehaviorptc/project_notebooks/ARSigmoidNormalHMM_2.ipynb)

    

  