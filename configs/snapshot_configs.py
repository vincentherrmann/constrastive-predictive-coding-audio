import datetime

snapshot_manager_default_dict = {
    'name': 'model_' + datetime.datetime.today().strftime('%Y-%m-%d') + '_run_0',
    'snapshot_location': '../snapshots',
    'logs_location': '../logs',
    'gcs_snapshot_location': 'snapshots',
    'gcs_logs_location': 'logs',
    'gcs_project': 'pytorch-wavenet',
    'gcs_bucket': 'immersions'
}