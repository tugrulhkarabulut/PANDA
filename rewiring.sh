echo "Running rewiring experiments..."

echo "Running DIGL..."
python run_graph_classification.py --rewiring=digl --dataset=reddit --layer_type=GCN --device=0 --num_trials=50 --alpha 0.15 --eps 1e-3
python run_graph_classification.py --rewiring=digl --dataset=imdb --layer_type=GCN --device=0 --num_trials=50 --alpha 0.05 --eps 1e-4
python run_graph_classification.py --rewiring=digl --dataset=mutag --layer_type=GCN --device=0 --num_trials=50 --alpha 0.15 --eps 1e-3
python run_graph_classification.py --rewiring=digl --dataset=enzymes --layer_type=GCN --device=0 --num_trials=50 --alpha 0.15 --eps 1e-4
python run_graph_classification.py --rewiring=digl --dataset=proteins --layer_type=GCN --device=0 --num_trials=50 --alpha 0.15 --eps 1e-4
python run_graph_classification.py --rewiring=digl --dataset=collab --layer_type=GCN --device=0 --num_trials=50 --alpha 0.05 --eps 1e-4


echo "Running SDRF..."
python run_graph_classification.py --rewiring=sdrf --dataset=reddit --layer_type=GCN --device=0 --num_trials=50 --num_iterations=5
python run_graph_classification.py --rewiring=sdrf --dataset=imdb --layer_type=GCN --device=0 --num_trials=50 --num_iterations=20
python run_graph_classification.py --rewiring=sdrf --dataset=mutag --layer_type=GCN --device=0 --num_trials=50 --num_iterations=5
python run_graph_classification.py --rewiring=sdrf --dataset=enzymes --layer_type=GCN --device=0 --num_trials=50 --num_iterations=5
python run_graph_classification.py --rewiring=sdrf --dataset=proteins --layer_type=GCN --device=0 --num_trials=50 --num_iterations=40
python run_graph_classification.py --rewiring=sdrf --dataset=collab --layer_type=GCN --device=0 --num_trials=50 --num_iterations=5


echo "Running FoSR..."
python run_graph_classification.py --rewiring=fosr --dataset=reddit --layer_type=GCN --device=0 --num_trials=50 --num_iterations=5
python run_graph_classification.py --rewiring=fosr --dataset=imdb --layer_type=GCN --device=0 --num_trials=50 --num_iterations=5
python run_graph_classification.py --rewiring=fosr --dataset=mutag --layer_type=GCN --device=0 --num_trials=50 --num_iterations=40
python run_graph_classification.py --rewiring=fosr --dataset=enzymes --layer_type=GCN --device=0 --num_trials=50 --num_iterations=10
python run_graph_classification.py --rewiring=fosr --dataset=proteins --layer_type=GCN --device=0 --num_trials=50 --num_iterations=20
python run_graph_classification.py --rewiring=fosr --dataset=collab --layer_type=GCN --device=0 --num_trials=50 --num_iterations=10