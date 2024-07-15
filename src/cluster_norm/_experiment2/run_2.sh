# # iterate over layers: 7, 23, 31, 15 in for loop instead
# for layer in 7 23 31 15
# do
#     for variant in 1 2
#     do
#         # python harvest.py phi $layer positive $variant
#         # python harvest.py phi $layer negative $variant
#     done
# done

# python harvest.py mistral 7 negative 1
# python harvest.py mistral 7 negative 2
# python harvest.py mistral 15 positive 1
# python harvest.py mistral 15 positive 2
# python harvest.py mistral 31 positive 1

# # method, model, layer, normalize = sys.argv[1:5]
# for layer in 23 15 31 7
# do 
#     for method in "lr" "crc" "ccs"
#     do
#         for model in "mistral" "phi" "llama"
#         do
#             for normalize in "burns" "cluster"
#             do
#                 python experiment.py $method $model $layer $normalize
#             done
#         done
#     done
# done


# python harvest.py meta-llama/Meta-Llama-3-8B 23 positive 1
# python harvest.py meta-llama/Meta-Llama-3-8B 23 negative 1
# python experiment.py


# python harvest.py mistral-7b 23 positive 1
# python harvest.py mistral-7b 23 negative 1

# python harvest.py gemma-7b 23 positive 1 

# python harvest.py "google/gemma-2-9b" 23 negative 1
# python harvest.py "google/gemma-2-9b" 23 positive 2 

python harvest.py "google/gemma-2-9b" 23 negative 2 
python harvest.py "google/gemma-2-9b" 23 positive 1

# python harvest.py phi-3 23 negative 2 
# python experiment.py
