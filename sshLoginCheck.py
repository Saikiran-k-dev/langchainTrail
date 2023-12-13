import paramiko

jumpServerHostname = '54.68.154.157'
jumpServerPort = 969  
jumpServerUsername = 'saikiran'

rvpnServerHostname = '20.0.2.35'
rvpnServerPort = 969  
rvpnServerUsername = 'devuser'

try:
    jumpServerSsh = paramiko.SSHClient()
    jumpServerSsh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    jumpServerSsh.connect(jumpServerHostname, jumpServerPort, jumpServerUsername)
    stdin, stdout, stderr = jumpServerSsh.exec_command('sudo ls -l')
    print(stdout.read().decode())

   

    rvpnServerSsh = paramiko.SSHClient()
    try:
        rvpnServerSsh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        rvpnServerSsh.connect(rvpnServerHostname, rvpnServerPort, rvpnServerUsername)
        stdin, stdout, stderr = rvpnServerSsh.exec_command("ls -l")
        print(stdout.read().decode())
        rvpnServerSsh.close()
    except Exception as e:
        print(e)
    jumpServerSsh.close()
except Exception as e:
    print(e)
