import time
import datetime
import threading

import psutil
import boto3
import nvgpu
import pynvml as nv

def get_gpu_mem_usage(gpu_id=0):
    gpu_infos = nvgpu.gpu_info()
    gpu_info = gpu_infos[gpu_id]
    gpu_mem_used_pct = gpu_info['mem_used_percent']

    return gpu_mem_used_pct


def get_gpu_util(gpu_id=0):
    nv.nvmlInit()

    handle = nv.nvmlDeviceGetHandleByIndex(gpu_id)
    print('AWS DEBUG nvmlDeviceGetHandleByIndex', handle)
    utilization = nv.nvmlDeviceGetUtilizationRates(handle).gpu
    print('AWS DEBUG nvmlDeviceGetUtilizationRates.gpu', utilization)

    return utilization


class GPUMon(threading.Thread):
    def __init__(self,
                 sleep_interval=1,
                 device_index=0,
                 job_name='training_job'):
        super().__init__()
        # Thread config
        self._kill = threading.Event()
        self._interval = sleep_interval

        # GPU monitoring config
        nv.nvmlInit()
        self.device_index = device_index
        self.handle = nv.nvmlDeviceGetHandleByIndex(device_index)

        self.cloudwatch = boto3.client('cloudwatch',
                                       region_name='us-east-1')
        self.job_name = job_name

    def run(self):
        while True:
            utilization = nv.nvmlDeviceGetUtilizationRates(self.handle).gpu
            gpu_mem_pct = get_gpu_mem_usage(self.device_index)
            datetime_object = datetime.datetime.now()
            cpu_pct = psutil.cpu_percent()
            ram_pct = psutil.virtual_memory().percent

            print(
                f'\n{datetime_object} - CPU {cpu_pct:5.2f}%, RAM util - {ram_pct:5.2f}%, GPU {self.device_index} util - {utilization:5.2f}%, GPU mem util - {gpu_mem_pct:5.2f}%'
            )
            
#             self.get_memory()

            self.logResults(cpu_pct, ram_pct, utilization, gpu_mem_pct)
            is_killed = self._kill.wait(self._interval)
            if is_killed:
                break

    def get_memory(self):
        virt = psutil.virtual_memory()
        swap = psutil.swap_memory()
        templ = "%-7s %10s %10s %10s %10s %10s %10s"
        print(templ % ('', 'total', 'used', 'free', 'shared', 'buffers', 'cache'))
        print(templ % (
            'Mem:',
            int(virt.total / 1024),
            int(virt.used / 1024),
            int(virt.free / 1024),
            int(getattr(virt, 'shared', 0) / 1024),
            int(getattr(virt, 'buffers', 0) / 1024),
            int(getattr(virt, 'cached', 0) / 1024)))
        print(templ % (
            'Swap:', int(swap.total / 1024),
            int(swap.used / 1024),
            int(swap.free / 1024),
            '',
            '',
            ''))
        
        
    def kill(self):
        self._kill.set()

    def logResults(self, cpu_util, ram_util, gpu_util, gpu_mem_util):
        store_reso = 1

        MY_DIMENSIONS = [{'Name': 'JobName', 'Value': self.job_name}]

        self.cloudwatch.put_metric_data(MetricData=[
            {
                'MetricName': 'GPU Usage',
                'Dimensions': MY_DIMENSIONS,
                'Unit': 'Percent',
                'StorageResolution': store_reso,
                'Value': gpu_util
            },
            {
                'MetricName': 'GPU Memory Usage',
                'Dimensions': MY_DIMENSIONS,
                'Unit': 'Percent',
                'StorageResolution': store_reso,
                'Value': gpu_mem_util
            },
            {
                'MetricName': 'CPU Usage',
                'Dimensions': MY_DIMENSIONS,
                'Unit': 'Percent',
                'StorageResolution': store_reso,
                'Value': cpu_util
            },
            {
                'MetricName': 'RAM Usage',
                'Dimensions': MY_DIMENSIONS,
                'Unit': 'Percent',
                'StorageResolution': store_reso,
                'Value': ram_util
            },
        ],
        Namespace='LGE-RD-AWS-Ultracluster')


def print_gpu_mem_pct(args, gpu_id, comment='NONE', verbose=False):

    if args.debug == True:
        gpu_mem_pct = get_gpu_mem_usage(args.local_rank)
        print(f'{comment} - GPU {gpu_id} memory usage : {gpu_mem_pct:.4f}%')

        #     gpu_mem_pct = get_gpu_mem_usage(1)
        #     print(f'{comment} - GPU 1 memory usage : {gpu_mem_pct:.4f} %')

        
