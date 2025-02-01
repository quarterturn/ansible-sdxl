# ansible-sdxl
run SDXL image generation using an Ansible playbook

a work in progress!

usage:
install ansible: sudo apt install ansible
install ansible-navigator: pip3 install ansible-navigator
create a conda environment: conda create -n ansible-sdxl python=3.11
activate the environment: conda activate ansible-sdxl
install the dependencies: pip3 install -r requirements.txt 
build and install the collection:
    ansible-galaxy collection build ansible_collections/my_namespace/stable_diffusion --output-path .
    ansible-galaxy collection build --force ansible_collections/my_namespace/stable_diffusion --output-path .
log into huggingface with a token: huggingface-cli login
run the playbook:
    ansible-playbook main.yml

to-do:
allow configuring model, vae, prompts, parameters etc... via variables
create an execution environment image
