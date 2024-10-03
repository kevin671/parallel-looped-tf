import torch


def parallel_inference(model, idx, parallel, memory_size, max_steps, stop_idx=-1):
    # Parallel inference with  Triangular Anderson acceleration

    # Picard iteration:
    # x^k+1 = f(x^k, t)
    # f(x^k, t) = x_0 + sum_{i=0}^{t} (TF(x^k, t) - x^k)

    # Anderson acceleration:
    # memory: the number of previous steps to remember
    # history_buffer = [x^k-M, x^k-M+1, ..., x^k-1]
    # x^k+1 = f(x^k) - G^k R^k where G^k = (I - R^k)^-1
    # x^k = [x^k[0], x^k[1], ..., x^k[L]]

    emb = model.transformer.embedding(idx)
    x = model.transformer.drop(emb)
    _, seq_len, hidden_dim = x.size()

    num_loops = model.num_loop

    timesteps = torch.arange(0, num_loops, device=x.device)
    parallel = parallel  # min(parallel, len(timesteps))
    begin_idx = 0
    end_idx = begin_idx + parallel

    latents_time_evolution_buffer = torch.stack([x] * (len(timesteps) + 1))
    residual_memory = None

    memory_indexes = torch.zeros(num_loops + 1, device=x.device, dtype=torch.long)

    logits_list = []
    tolerance = 1.0
    stop_criteria = tolerance + 1.0
    n = 0
    while stop_criteria > tolerance and len(logits_list) < max_steps:
        print("step", n)
        n += 1
        # print(begin_idx, end_idx)
        parallel_len = end_idx - begin_idx
        block_latents = latents_time_evolution_buffer[begin_idx:end_idx]  # x^k
        t_vec = timesteps[begin_idx:end_idx]

        model_output = torch.zeros_like(block_latents)
        for _i, _t in enumerate(t_vec):
            with torch.no_grad():
                print(_i)
                print(block_latents[_i][0, stop_idx, :50])
                model_output[_i] = (
                    model.f(block_latents[_i], _t.item()) - block_latents[_i]
                )
                print(model_output[_i][0, stop_idx, :50])
        delta = model_output.reshape(parallel_len, 1, seq_len, hidden_dim)
        cumulative_delta = torch.cumsum(delta, dim=0)
        block_latents_new = (
            latents_time_evolution_buffer[begin_idx][None,] + cumulative_delta
        )  # f(x^k)

        # debug
        last_latent = block_latents_new[-1]
        print(last_latent[0, stop_idx, :50])
        _x = model.transformer.ln_f(last_latent)
        _logits = model.lm_head(_x)
        _pred = torch.argmax(_logits[0, stop_idx])
        print("pred", _pred)

        # f(x^k) - x^k
        cur_error_vec = (
            block_latents_new
            - latents_time_evolution_buffer[begin_idx + 1 : end_idx + 1]
        )  # [parallel_len, 1, seq_len, hidden_dim]
        ho_residual = cur_error_vec.to(
            torch.float64
        )  # [parallel_len, 1, seq_len, hidden_dim] # R
        cur_error = torch.linalg.vector_norm(
            cur_error_vec[:, 0, stop_idx, :], dim=-1
        )  # [parallel_len]
        print(cur_error)

        # Anderson acceleration
        Gf = torch.zeros_like(ho_residual)

        if residual_memory is None:
            residual_memory = torch.zeros(
                1,
                num_loops + 1,
                1,
                seq_len,
                hidden_dim,
                device=x.device,
                dtype=torch.float64,
            )
            samples_memory = torch.zeros(
                1,
                num_loops + 1,
                1,
                seq_len,
                hidden_dim,
                device=x.device,
                dtype=torch.float64,
            )
            residual_memory[0, t_vec] = ho_residual
            samples_memory[0, t_vec] = latents_time_evolution_buffer[
                begin_idx + 1 : end_idx + 1
            ].to(torch.float64)
            memory_indexes[t_vec] = torch.clamp(
                memory_indexes[t_vec] + 1, max=memory_size
            )
        else:
            padded_residual = torch.zeros(
                1,
                num_loops + 1,
                1,
                seq_len,
                hidden_dim,
                device=x.device,
                dtype=torch.float64,
            )
            padded_samples = torch.zeros(
                1,
                num_loops + 1,
                1,
                seq_len,
                hidden_dim,
                device=x.device,
                dtype=torch.float64,
            )
            padded_residual[0, t_vec] = ho_residual
            padded_samples[0, t_vec] = latents_time_evolution_buffer[
                begin_idx + 1 : end_idx + 1
            ].to(torch.float64)

            residual_memory = torch.cat([residual_memory, padded_residual], dim=0)
            samples_memory = torch.cat([samples_memory, padded_samples], dim=0)

            residual_memory = residual_memory[-memory_size:]
            samples_memory = samples_memory[-memory_size:]
            memory_indexes[t_vec] = torch.clamp(
                memory_indexes[t_vec] + 1, max=memory_size
            )

            residual_diff = residual_memory[1:] - residual_memory[:-1]
            sample_diff = samples_memory[1:] - samples_memory[:-1]

            residual_diff_t = residual_diff[:, t_vec, :, :, :]
            sample_diff_t = sample_diff[:, t_vec, :, :, :]  #

            # print("residual_diff", residual_diff.shape, "sample_diff", sample_diff.shape)

            use_memory = memory_indexes[t_vec]  # [parallel_len]
            use_memory_max = (
                use_memory.max()
            )  # memory_size  # min(memory_size, len(t_vec))
            sample_diff_mat = sample_diff_t[
                :use_memory_max, :, :, :, :
            ]  # [m_k, parallel_len, 1, seq_len, hidden_dim]
            res_diff_mat = residual_diff_t[
                :use_memory_max, :, :, :, :
            ]  # [m_k, parallel_len, 1, seq_len, hidden_dim]
            flip_res_diff_mat = torch.flip(res_diff_mat, dims=[1])
            B = torch.einsum(
                "ijklm,pjklm->ipj", flip_res_diff_mat, flip_res_diff_mat
            )  # [m_k, m_k, parallel_len]
            B = torch.flip(torch.cumsum(B, dim=2), dims=[2])  #
            B = B.permute(2, 0, 1)  # [parallel_len, m_k, m_k]

            flip_ho_residual = torch.flip(ho_residual, dims=[0])
            d = torch.einsum(
                "ijklm,jklm->ji", flip_res_diff_mat, flip_ho_residual
            )  # [m_k, parallel_len]
            d = torch.flip(torch.cumsum(d, dim=0), dims=[0])  # [parallel_len, m_k]

            ind = torch.argmax((cur_error > tolerance).int()).item() + 1
            # print(torch.arange(d.shape[1], device = d.device).unsqueeze(0).shape) # [1, m_k]
            # indices = torch.arange(d.shape[1], device = d.device).unsqueeze(0).expand(d.shape)
            indices = (
                torch.arange(ind, device=d.device).unsqueeze(0).expand(d.shape[0], ind)
            )
            mask_d = indices
            # print(indices)
            # print(indices.shape, mask_d.shape)
            d[mask_d] = 0
            # print(d)

            # mask_B = mask_d.unsqueeze(2) | mask_d.unsqueeze(1)
            # B[mask_B] = 0
            # print(B.shape, d.shape)
            # B = B + 1e3 * torch.eye(use_memory_max, device=B.device, dtype = torch.float64).unsqueeze(0) # [parallel_len, m_k, m_k]
            # print(B.shape, d.shape)

            if use_memory_max == 1:
                solve_d = d / B.squeeze(-1)
            else:
                solve_d = torch.linalg.solve(B, d)  # [parallel_len, m_k]

            # print(solve_d)

            A = (
                sample_diff_mat + res_diff_mat
            )  # [m_k, parallel_len, 1, seq_len, hidden_dim]

            # print("debug", A.shape, solve_d.shape) # [m_k, parallel_len, 1, seq_len, hidden_dim], [parallel_len, m_k]
            Gf_flat = torch.einsum(
                "ijklm,ji->jklm", A, solve_d
            )  # [parallel_len, 1, seq_len, hidden_dim]
            Gf = Gf_flat

        # update x^k
        latents_time_evolution_buffer[begin_idx + 1 : end_idx + 1] = (
            block_latents_new - Gf
        )

        # post-processing

        # stop criterion: ||x^k - x^k-1||_2
        # cur_error = torch.linalg.vector_norm(cur_error_vec[:, 0, stop_idx, :], dim=-1)  # [parallel_len]
        # print(cur_error)
        stop_criteria = cur_error[-1]

        # approximation logits at k-th step
        with torch.no_grad():
            logits = model.lm_head(model.transformer.ln_f(block_latents_new[-1]))
            logits_list.append(logits)

    return torch.stack(logits_list, dim=0)
