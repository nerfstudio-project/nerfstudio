#include "utils.h"


std::vector<torch::Tensor> ray_aabb_intersect(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor centers,
    const torch::Tensor half_sizes,
    const int max_hits
){
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(centers);
    CHECK_INPUT(half_sizes);
    return ray_aabb_intersect_cu(rays_o, rays_d, centers, half_sizes, max_hits);
}


std::vector<torch::Tensor> ray_sphere_intersect(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor centers,
    const torch::Tensor radii,
    const int max_hits
){
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(centers);
    CHECK_INPUT(radii);
    return ray_sphere_intersect_cu(rays_o, rays_d, centers, radii, max_hits);
}


void packbits(
    torch::Tensor density_grid,
    const float density_threshold,
    torch::Tensor density_bitfield
){
    CHECK_INPUT(density_grid);
    CHECK_INPUT(density_bitfield);
    
    return packbits_cu(density_grid, density_threshold, density_bitfield);
}


torch::Tensor morton3D(const torch::Tensor coords){
    CHECK_INPUT(coords);

    return morton3D_cu(coords);
}


torch::Tensor morton3D_invert(const torch::Tensor indices){
    CHECK_INPUT(indices);

    return morton3D_invert_cu(indices);
}


std::vector<torch::Tensor> raymarching_train(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor hits_t,
    const torch::Tensor density_bitfield,
    const float scale,
    const float exp_step_factor,
    const torch::Tensor noise,
    const int grid_size,
    const int max_samples
){
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(hits_t);
    CHECK_INPUT(density_bitfield);
    CHECK_INPUT(noise);

    return raymarching_train_cu(
        rays_o, rays_d, hits_t, density_bitfield, scale, exp_step_factor,
        noise, grid_size, max_samples);
}


std::vector<torch::Tensor> raymarching_test(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    torch::Tensor hits_t,
    const torch::Tensor alive_indices,
    const torch::Tensor density_bitfield,
    const float scale,
    const float exp_step_factor,
    const int grid_size,
    const int max_samples,
    const int N_samples
){
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(hits_t);
    CHECK_INPUT(alive_indices);
    CHECK_INPUT(density_bitfield);

    return raymarching_test_cu(
        rays_o, rays_d, hits_t, alive_indices, density_bitfield, scale, exp_step_factor,
        grid_size, max_samples, N_samples);
}


std::vector<torch::Tensor> composite_train_fw(
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const float opacity_threshold
){
    CHECK_INPUT(sigmas);
    CHECK_INPUT(rgbs);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(rays_a);

    return composite_train_fw_cu(
                sigmas, rgbs, deltas, ts,
                rays_a, opacity_threshold);
}


std::vector<torch::Tensor> composite_train_bw(
    const torch::Tensor dL_dopacity,
    const torch::Tensor dL_ddepth,
    const torch::Tensor dL_drgb,
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const torch::Tensor opacity,
    const torch::Tensor depth,
    const torch::Tensor rgb,
    const float opacity_threshold
){
    CHECK_INPUT(dL_dopacity);
    CHECK_INPUT(dL_ddepth);
    CHECK_INPUT(dL_drgb);
    CHECK_INPUT(sigmas);
    CHECK_INPUT(rgbs);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(rays_a);
    CHECK_INPUT(opacity);
    CHECK_INPUT(depth);
    CHECK_INPUT(rgb);

    return composite_train_bw_cu(
                dL_dopacity, dL_ddepth, dL_drgb,
                sigmas, rgbs, deltas, ts, rays_a,
                opacity, depth, rgb, opacity_threshold);
}


void composite_test_fw(
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor hits_t,
    const torch::Tensor alive_indices,
    const float T_threshold,
    const torch::Tensor N_eff_samples,
    torch::Tensor opacity,
    torch::Tensor depth,
    torch::Tensor rgb
){
    CHECK_INPUT(sigmas);
    CHECK_INPUT(rgbs);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(hits_t);
    CHECK_INPUT(alive_indices);
    CHECK_INPUT(N_eff_samples);
    CHECK_INPUT(opacity);
    CHECK_INPUT(depth);
    CHECK_INPUT(rgb);

    composite_test_fw_cu(
        sigmas, rgbs, deltas, ts, hits_t, alive_indices,
        T_threshold, N_eff_samples,
        opacity, depth, rgb);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("ray_aabb_intersect", &ray_aabb_intersect);
    m.def("ray_sphere_intersect", &ray_sphere_intersect);

    m.def("morton3D", &morton3D);
    m.def("morton3D_invert", &morton3D_invert);
    m.def("packbits", &packbits);

    m.def("raymarching_train", &raymarching_train);
    m.def("raymarching_test", &raymarching_test);
    m.def("composite_train_fw", &composite_train_fw);
    m.def("composite_train_bw", &composite_train_bw);
    m.def("composite_test_fw", &composite_test_fw);

}