#define MUL_MV_ID_LOWBIT_ARGS \
        global char * src1, \
        ulong offset1, \
        global const char * ids, \
        ulong offset_ids, \
        global char * dst, \
        ulong offsetd, \
        int ne00, \
        int ne01, \
        ulong nb01, \
        ulong nb02, \
        int ne11, \
        ulong nb11, \
        ulong nb12, \
        ulong nb_ids, \
        ulong nbd1, \
        ulong nbd2

inline void mul_mv_id_lowbit_zero(global float * dst, int first_row, int n_rows, int rows_per_group) {
    const int row = (int) get_local_id(0);
    if (row < rows_per_group && first_row + row < n_rows) {
        dst[first_row + row] = 0.0f;
    }
}
