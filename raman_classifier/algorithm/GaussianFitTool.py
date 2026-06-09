class GaussianFitTool:

    _init_thread = None
    _process_pool = None

    @staticmethod
    def _fit_single_curve(x, y, i):
        import numpy as np
        from scipy.optimize import minimize
        from scipy.signal import find_peaks

        x_min = x[0]
        x_max = x[-1]
        dx = x[1] - x[0]
        x_span = x_max - x_min

        peaks, properties = find_peaks(y, height=1, prominence=0.1, width=True)
        n_peaks = len(peaks)

        widths = properties["widths"]
        peak_heights = properties["peak_heights"]

        mu = x[peaks]
        sigma = widths * dx / 2.35482
        amp = peak_heights

        params0 = np.concatenate([
            amp, mu, sigma, np.zeros((1,))
        ])

        params_lb = np.concatenate([
            np.full((n_peaks,), 0),
            np.full((n_peaks,), x_min),
            np.full((n_peaks,), dx),
            np.zeros((1,))
        ])
        params_ub = np.concatenate([
            np.ones((n_peaks,)),
            np.full((n_peaks,), x_max),
            np.full((n_peaks,), x_span / 3),
            np.ones((1,))
        ])

        result = minimize(GaussianFitTool._object_func, x0=params0, jac=True, args=(x, y), bounds=list(zip(params_lb, params_ub)))
        params = result.x
        
        peaks_amp = params[:n_peaks]
        try:
            max_amp = peaks_amp.max()
        except:
            max_amp = 0

        peaks_mu = params[n_peaks:2*n_peaks]
        peaks_sigma = params[2*n_peaks:3*n_peaks]
        peaks_offset = params[-1:]

        if max_amp > 0:
            mask = (
                (peaks_amp/max_amp >= 0.3) &
                (peaks_mu - peaks_sigma >= x_min) &
                (peaks_mu + peaks_sigma <= x_max) &
                ((peaks_amp/max_amp) / peaks_sigma >= 0.1)
            )

            is_changed = not np.all(mask)
            if is_changed:
                peaks_amp = peaks_amp[mask]
                peaks_mu = peaks_mu[mask]
                peaks_sigma = peaks_sigma[mask]

            if not np.all(np.diff(peaks_mu) >= 0):
                sorted_indices = np.argsort(peaks_mu)
                peaks_amp = peaks_amp[sorted_indices]
                peaks_mu = peaks_mu[sorted_indices]
                peaks_sigma = peaks_sigma[sorted_indices]
                is_changed:bool = True

            if is_changed:
                params = np.concatenate([peaks_amp, peaks_mu, peaks_sigma, peaks_offset])
        else:
            params = peaks_offset

        fitted_y = GaussianFitTool._multiple_gaussian(x, *params)

        return params, fitted_y, i

    @staticmethod
    def _object_func(params, x, y):
        import numpy as np

        n_peaks = (params.shape[0] - 1) // 3
        n_cols = x.size
        
        peaks_amp = params[0 : n_peaks]
        peaks_mu = params[n_peaks : 2*n_peaks]
        peaks_sigma = params[2*n_peaks : 3*n_peaks]
        peaks_offset = params[3*n_peaks]
        peaks_sigma2 = peaks_sigma**2
        peaks_sigma3 = peaks_sigma**3

        X = np.broadcast_to(x, (n_peaks, n_cols)).T
        diff = X - peaks_mu
        diff2 = diff ** 2
        exp_terms = np.exp(-diff2 / (2*peaks_sigma2)) # (n_cols, n_peaks)
        Y_pred = peaks_amp * exp_terms
        y_pred = Y_pred.sum(axis=1) + peaks_offset

        residual = y - y_pred
        cost = np.sum(residual**2)

        grad = np.zeros_like(params)
        
        grad[:n_peaks] = residual @ exp_terms
        
        dy_dmu = peaks_amp * exp_terms * diff / peaks_sigma2
        grad[n_peaks:2*n_peaks] = residual @ dy_dmu
        
        dy_dsigma = peaks_amp * exp_terms * diff2 / peaks_sigma3
        grad[2*n_peaks:3*n_peaks] = residual @ dy_dsigma
        
        grad[3*n_peaks] = residual.sum()

        cost = 100 * cost / n_cols
        grad = -200 * grad / n_cols

        return cost, grad
    
    @staticmethod
    def _multiple_gaussian(x, *params):
        import numpy as np

        n_peaks = (len(params) - 1) // 3
        offset = params[-1]
        
        params_array = np.array(params[:-1])
        amps = params_array[:n_peaks]
        mus  = params_array[n_peaks:2*n_peaks]
        sigmas = params_array[2*n_peaks:]

        X = np.broadcast_to(x, (n_peaks, x.shape[0])).T
        Y = amps * np.exp(-(X - mus)**2 / (2 * sigmas**2))
        return Y.sum(axis=1) + offset

    @staticmethod
    def fit(x, Y, stage=None):
        import time
        from concurrent.futures import wait, FIRST_COMPLETED

        GaussianFitTool.start(wait=True)

        print(f"gaussian fit start")
        start_time = time.perf_counter()
        
        not_done_futures = []
        total_rows = Y.shape[0]
        for i in range(total_rows):
            future = GaussianFitTool._process_pool.submit(
                GaussianFitTool._fit_single_curve, x, Y[i, :], i
            )
            not_done_futures.append(future)

        results = []
        while not_done_futures:
            done_futures, not_done_futures = wait(not_done_futures, return_when=FIRST_COMPLETED)
            results.extend(future.result() for future in done_futures)
            stage.progress = len(results)/total_rows

        results.sort(key=lambda x: x[2])

        stop_time = time.perf_counter()
        print("gaussian fit time elapsed:", stop_time - start_time)
        return results

    @staticmethod
    def _dummy(x):
        import time
        time.sleep(0.1)

    @staticmethod
    def _init():
        import psutil
        from concurrent.futures import ProcessPoolExecutor

        n_cores = psutil.cpu_count()
        GaussianFitTool._process_pool = ProcessPoolExecutor(max_workers=n_cores)
        list(GaussianFitTool._process_pool.map(GaussianFitTool._dummy, range(n_cores)))

    @staticmethod
    def start(wait=False):
        if GaussianFitTool._process_pool is not None:
            return

        if GaussianFitTool._init_thread is None:
            import threading
            GaussianFitTool._init_thread = threading.Thread(target=GaussianFitTool._init)
            GaussianFitTool._init_thread.start()

        if wait and GaussianFitTool._init_thread.is_alive():
            GaussianFitTool._init_thread.join()
            GaussianFitTool._init_thread = None

    @staticmethod
    def stop():
        if GaussianFitTool._init_thread is not None:
            if GaussianFitTool._init_thread.is_alive():
                GaussianFitTool._init_thread.join()

            GaussianFitTool._init_thread = None

        if GaussianFitTool._process_pool is not None:
            GaussianFitTool._process_pool.shutdown(wait=True)
            GaussianFitTool._process_pool = None