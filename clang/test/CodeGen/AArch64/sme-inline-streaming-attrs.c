// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -S -o /dev/null -target-feature +sme -target-feature +sme2 -verify -DTEST_NONE %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -S -o /dev/null -target-feature +sme -target-feature +sme2 -verify -DTEST_COMPATIBLE %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -S -o /dev/null -target-feature +sme -target-feature +sme2 -verify -DTEST_STREAMING %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -S -o /dev/null -target-feature +sme -target-feature +sme2 -verify -DTEST_LOCALLY %s

// REQUIRES: aarch64-registered-target

#define __ai __attribute__((always_inline))
__ai void inlined_fn(void) {}
__ai void inlined_fn_streaming_compatible(void) __arm_streaming_compatible {}
__ai void inlined_fn_streaming(void) __arm_streaming {}
__ai __arm_locally_streaming void inlined_fn_local(void) {}
__ai __arm_new("za") void inlined_fn_za(void) {}
__ai __arm_new("zt0") void inlined_fn_zt0(void) {}

#ifdef TEST_NONE
void caller(void) {
    inlined_fn();
    inlined_fn_streaming_compatible();
    inlined_fn_streaming(); // expected-error {{always_inline function 'inlined_fn_streaming' and its caller 'caller' have mismatching streaming attributes}}
    inlined_fn_local(); // expected-error {{always_inline function 'inlined_fn_local' and its caller 'caller' have mismatching streaming attributes}}
    inlined_fn_za(); // expected-error {{always_inline function 'inlined_fn_za' has new za state}}
    inlined_fn_zt0(); // expected-error {{always_inline function 'inlined_fn_zt0' has new zt0 state}}
}
#endif

#ifdef TEST_COMPATIBLE
void caller_compatible(void) __arm_streaming_compatible {
    inlined_fn(); // expected-warning {{always_inline function 'inlined_fn' and its caller 'caller_compatible' have mismatching streaming attributes, inlining may change runtime behaviour}}
    inlined_fn_streaming_compatible();
    inlined_fn_streaming(); // expected-error {{always_inline function 'inlined_fn_streaming' and its caller 'caller_compatible' have mismatching streaming attributes}}
    inlined_fn_local(); // expected-error {{always_inline function 'inlined_fn_local' and its caller 'caller_compatible' have mismatching streaming attributes}}
}
#endif

#ifdef TEST_STREAMING
void caller_streaming(void) __arm_streaming {
    inlined_fn(); // expected-warning {{always_inline function 'inlined_fn' and its caller 'caller_streaming' have mismatching streaming attributes, inlining may change runtime behaviour}}
    inlined_fn_streaming_compatible();
    inlined_fn_streaming();
    inlined_fn_local();
}
#endif

#ifdef TEST_LOCALLY
__arm_locally_streaming
void caller_local(void) {
    inlined_fn(); // expected-warning {{always_inline function 'inlined_fn' and its caller 'caller_local' have mismatching streaming attributes, inlining may change runtime behaviour}}
    inlined_fn_streaming_compatible();
    inlined_fn_streaming();
    inlined_fn_local();
}
#endif
