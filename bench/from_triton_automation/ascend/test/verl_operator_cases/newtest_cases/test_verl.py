from pathlib import Path

from UniAutos.TestEngine.Case import Case
from UniAutos.Util.TestStatus import TEST_STATUS

from lib.common.Sftp import ClassSftp
from lib.common.log.LogFactory import LogFactory
from lib.common.log.LogInterface import LOG_LEVEL_VERBOSE, LOG_TYPE_INFO, LOG_TYPE_ERROR
from lib.common.CLI import ClassCLI


# 非root用户source root用户的cann包环境变量，然后安装atb，安装失败
class CANN_TRITON_ACCU_VERL(Case):
    # 预置条件
    def preTestCase(self):
        self.plog = LogFactory.getLogger(self)
        self.plog.setLogLevel(LOG_LEVEL_VERBOSE)

        self.host = self.resource.getDevice(deviceType='host', deviceId='1')
        self.ip = self.host.rawParams['ipv4_address']
        self.username = self.host.username
        self.password = self.host.password
        self.port = 22

        # host
        self.ssh_connect_host = ClassCLI(self.ip, self.username, self.password)
        self.sftp = ClassSftp(self.ip, self.username, self.password)
        if not self.ssh_connect_host.login() or not self.sftp.login():
            self.plog.log(LOG_TYPE_ERROR, 'host ssh connect failed!'.center(20, "*"))
            self.setCaseStatus(TEST_STATUS.FAILED)
        else:
            self.plog.log(LOG_TYPE_INFO, 'host ssh connect successfully!')

    def check_directory_exist(self, file_path):
        """
        @Func：查询目录是否存在
        @Param file_path：文件
        @Return: True or False
        """
        cmd = "test -d {}; echo $?".format(file_path)
        result = self.ssh_connect_host.sshCmd(cmd, waitstr='root@')
        if result[0] == "0":
            return True
        return False

    # 测试步骤
    def procedure(self):
        # 依赖 CANN 8.5.0 triton-ascend==3.2.0
        self.ssh_connect_host.sshCmd('cd /home/;git clone https://gitcode.com/Ascend/triton-ascend-kernels.git')
        self.ssh_connect_host.sshCmd('cd /home/;git clone https://gitcode.com/Ascend/triton-ascend-kernels.git')
        self.ssh_connect_host.sshCmd('source /usr/local/Ascend/cann-8.5.0/set_env.sh')
        self.ssh_connect_host.sshCmd(
            'conda activate triton;cd triton-ascend-kernels;rm -f verl_log.txt;git reset --hard;git pull;'
            'pip3 install -e .')
        curr_dir = Path(__file__).resolve().parent
        self.sftp.putFile(str(curr_dir / 'test_verl_linear_cross_entropy.py'),
                          '/home/triton-ascend-kernels/tests/loss/test_verl_linear_cross_entropy.py')
        self.sftp.putFile(str(curr_dir / 'verl_cross_entropy_kernels.py'),
                          '/home/triton-ascend-kernels/src/triton_ascend_kernels/loss/verl_cross_entropy_kernels.py')
        self.ssh_connect_host.sshCmd('export ASCEND_RT_VISIBLE_DEVICES=15;'
                                     'pytest tests/loss/test_verl_linear_cross_entropy.py >> verl_log.txt')
        for i, backward in enumerate(['_Total_Fuse_MN', '_Total_Separate', '_Split_Dlogits_N']):
            self.ssh_connect_host.sshCmd(
                f"sed -i 's/_backward: BackwardEnum = BackwardEnum.*$/"
                f"_backward: BackwardEnum = BackwardEnum.{backward}/' "
                "src/triton_ascend_kernels/loss/verl_cross_entropy_kernels.py")
            if i == 0:
                self.ssh_connect_host.sshCmd('pytest tests/loss/test_verl_linear_cross_entropy.py >> verl_log.txt')
            else:
                self.ssh_connect_host.sshCmd(
                    'pytest tests/loss/test_verl_linear_cross_entropy.py::test_accuracy_linear_cross_entropy_bwd '
                    '>> verl_log.txt')
        ret = self.ssh_connect_host.sshCmd('grep FAILED -r verl_log.txt')
        if ret is not None:
            print(ret)
            self.plog.log(LOG_TYPE_ERROR, 'verl test failed!')
            self.setCaseStatus(TEST_STATUS.FAILED)
            return
        else:
            self.plog.log(LOG_TYPE_ERROR, 'verl test pass!')

    # 恢复环境
    def postTestCase(self):
        self.ssh_connect_host.close()
