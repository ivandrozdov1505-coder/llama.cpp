kernel void kernel_situ_f32(
        global const char * input,
        ulong gate_offset, ulong up_offset,
        global char * output, ulong out_offset,
        ulong width, ulong stride1, ulong stride2,
        float gate_scale, float gate_beta,
        float up_scale, float up_beta) {
    const size_t column = get_global_id(0);
    const size_t expert = get_global_id(1);
    const size_t token = get_global_id(2);
    if (column >= width) {
        return;
    }

    const ulong row = expert*stride1 + token*stride2;
    const float gate = ((global const float *) (input + gate_offset + row))[column];
    float up = ((global const float *) (input + up_offset + row))[column];
    if (up_beta > 0.0f) {
        up = tanh(up*up_scale)*up_beta;
    }

    const float sigmoid = 1.0f/(1.0f + exp(-gate));
    const float activation = (tanh(gate*gate_scale)*gate_beta)*sigmoid;
    global float * dst = (global float *) (output + out_offset);
    dst[(token*get_global_size(1) + expert)*width + column] = activation*up;
}
