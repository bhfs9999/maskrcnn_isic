from .cervix_eval import do_cervix_evaluation


def cervix_evaluation(dataset,
                      predictions,
                      output_folder,
                      box_only,
                      iou_types,
                      expected_results,
                      expected_results_sigma_tol,
                      draw
                      ):
    return do_cervix_evaluation(
        dataset=dataset,
        predictions=predictions,
        box_only=box_only,
        output_folder=output_folder,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
        draw=draw
    )
