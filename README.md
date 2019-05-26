# ngraph-rs
Rust bindings for nGraph

Work in Progress!!!

# Dependencies
See https://github.com/NervanaSystems/ngraph and https://ngraph.nervanasys.com/docs/latest/buildlb.html

Either install ngraph to system directories (ie sudo make install), or set environmental variables for header files and libraries.

For example, on Ubuntu, add the following to ~/.bashrc:
export $CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:$HOME/Documents/ngraph/src/"
export $LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/Documents/ngraph/build/ngraph/src" 

# Cargo
Add the following to cargo.toml:
ngraph = { git = "https://github.com/charles-r-earp/ngraph-rs" }
