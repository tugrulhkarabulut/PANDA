python run_graph_classification.py --rewiring=panda --dataset=reddit-binary --layer_type=PANDA-GCN --exp_factor=1.25 --centrality=degree --top_k=7 --device=0
python run_graph_classification.py --rewiring=panda --dataset=imdb-binary --layer_type=PANDA-GCN --exp_factor=1.25 --centrality=pagerank --top_k=3 --device=0
python run_graph_classification.py --rewiring=panda --dataset=mutag --layer_type=PANDA-GCN --exp_factor=1.75 --centrality=betweenness --top_k=10 --device=0
python run_graph_classification.py --rewiring=panda --dataset=enzymes --layer_type=PANDA-GCN --exp_factor=2 --centrality=pagerank --top_k=7 --device=0
python run_graph_classification.py --rewiring=panda --dataset=proteins --layer_type=PANDA-GCN --exp_factor=2 --centrality=closeness --top_k=7 --device=0
python run_graph_classification.py --rewiring=panda --dataset=collab --layer_type=PANDA-GCN --exp_factor=2 --centrality=degree --top_k=3 --device=0