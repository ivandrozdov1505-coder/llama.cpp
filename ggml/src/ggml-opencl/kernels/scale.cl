#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void kernel_scale_f32(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        float scale,
        float bias
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);
    dst[get_global_id(0)] = src0[get_global_id(0)] * scale + bias;
}

kernel void kernel_scale_f32_4(
        global float4 * src0,
        ulong offset0,
        global float4 * dst,
        ulong offsetd,
        float scale,
        float bias
) {
    src0 = (global float4*)((global char*)src0 + offset0);
    dst = (global float4*)((global char*)dst + offsetd);
    dst[get_global_id(0)] = src0[get_global_id(0)] * scale + bias;
}

kernel void kernel_scale_f32_nc(
        global const char * src, ulong src_offset,
        global char * dst, ulong dst_offset,
        float scale, float bias,
        ulong width,
        ulong src_stride1, ulong src_stride2, ulong src_stride3,
        ulong dst_stride1, ulong dst_stride2, ulong dst_stride3) {
    const size_t i1 = get_group_id(0);
    const size_t i2 = get_group_id(1);
    const size_t i3 = get_group_id(2);
    global const float * x = (global const float *) (src + src_offset + i1*src_stride1 + i2*src_stride2 + i3*src_stride3);
    global float * y = (global float *) (dst + dst_offset + i1*dst_stride1 + i2*dst_stride2 + i3*dst_stride3);
    for (size_t i = get_local_id(0); i < width; i += get_local_size(0)) {
        y[i] = x[i]*scale + bias;
    }
}
