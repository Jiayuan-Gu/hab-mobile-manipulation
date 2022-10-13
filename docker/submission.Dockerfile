FROM fairembodied/habitat-challenge:habitat_rearrangement_2022_base_docker

RUN /bin/bash -c "source activate habitat; pip install 'lmdb>=0.98' webdataset==0.1.40 ifcfg 'moviepy>=1.0.1' torch==1.5.1 tensorboard pybullet==3.0.4; pip cache purge"

ADD habitat-challenge/configs/ /configs/

ENV AGENT_EVALUATION_TYPE remote
ENV TRACK_CONFIG_FILE "/configs/tasks/rearrange_easy.local.rgbd.yaml"

ADD habitat_extensions /solution/habitat_extensions
ADD mobile_manipulation /solution/mobile_manipulation
ADD data/results/rearrange/skills/challenge/models /solution/models

CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/solution:/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && python solution/mobile_manipulation/challenge_agent.py --evaluation $AGENT_EVALUATION_TYPE --models-dir /solution/models"]