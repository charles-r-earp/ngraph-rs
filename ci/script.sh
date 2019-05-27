# This script takes care of testing your crate

set -ex

# TODO This is the "test phase", tweak it as you see fit
main() {
    ngraph build --target $TARGET
    ngraph build --target $TARGET --release

    if [ ! -z $DISABLE_TESTS ]; then
        return
    fi

    ngraph test --target $TARGET
    ngraph test --target $TARGET --release

    ngraph run --target $TARGET
    ngraph run --target $TARGET --release
}

# we don't run the "test phase" when doing deploys
if [ -z $TRAVIS_TAG ]; then
    main
fi
