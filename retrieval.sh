# retrieval batchsize2000のELSAをretrievalタスクで評価（AudioCapsのtest でセット)

python3 evaluate/retrieval.py   --checkpoint /home/takamichi-lab-pc09/elsa/ckpt/takamichi09/elsa_ckpt_batch2000/epoch_009.pt   --csv /home/takamichi-lab-pc09/elsa/Spatial_AudioCaps/AudioCaps_csv/test.csv   --audio_dir /home/takamichi-lab-pc09/elsa/Spatial_AudioCaps/takamichi09/AudioCaps_mp3/test   --batch_size 64 

#SpatialAudioCapsのtestでセット
(.venv) takamichi-lab-pc09@takamichi-lab-pc09:~/elsa$ python3 retrieval_spatial.py   --checkpoint /home/takamichi-lab-pc09/elsa/ckpt/takamichi09/elsa_ckpt_batch2000/epoch_009.pt   --csv /home/takamichi-lab-pc09/elsa/manifest_test.csv   --audio_dir /home/takamichi-lab-pc09/elsa/Spatial_AudioCaps/takamichi09/SpatialAudioCaps/foa/tes
t   --batch_size 64




# ELSAの音の埋め込みについてクエリから近いのを取得する.
python3 audio_embd_query_umap.py \
  --checkpoint /home/takamichi-lab-pc09/elsa/ckpt/takamichi09/elsa_ckpt_batch2000/epoch_009.pt\
  --csv /home/takamichi-lab-pc09/elsa/manifest_test.csv  \
  --audio_dir /home/takamichi-lab-pc09/elsa/Spatial_AudioCaps/takamichi09/SpatialAudioCaps/foa/test \
  --query_index 100



(.venv) takamichi-lab-pc09@takamichi-lab-pc09:~/elsa$ python3 audio_embd_query.py   --checkpoint /home/takamichi-lab-pc09/elsa/ckpt/takamichi09/elsa_ckpt_batch2000/epoch_009.pt  --csv /home/takamichi-lab-pc09/elsa/manifest_test.csv    --audio_dir /home/takamichi-lab-pc09/elsa/Spatial_AudioCaps/takamichi09/SpatialAudioCaps/foa/test   --query_index 100