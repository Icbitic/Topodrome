"""Actual SSH channels, with a stdlib receiver and no file-transfer subsystem."""

import asyncio
import shlex

import asyncssh

from topodrome.deployment.ssh_files import upload_bytes, upload_file


class _Server(asyncssh.SSHServer):
    def begin_auth(self, _username):
        return False


async def test_administrative_installation_streams_over_actual_ssh(tmp_path):
    async def execute(process):
        child = await asyncio.create_subprocess_exec(
            *shlex.split(process.command),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def feed():
            while chunk := await process.stdin.read(256 * 1024):
                child.stdin.write(chunk)
                await child.stdin.drain()
            child.stdin.close()

        async def output(source, target):
            while chunk := await source.read(256 * 1024):
                target.write(chunk)
                await target.drain()

        try:
            await asyncio.gather(
                feed(), output(child.stdout, process.stdout), output(child.stderr, process.stderr)
            )
            process.exit(await child.wait())
        finally:
            if child.returncode is None:
                child.terminate()
                await child.wait()

    server = await asyncssh.create_server(
        _Server,
        "127.0.0.1",
        0,
        server_host_keys=[asyncssh.generate_private_key("ssh-ed25519")],
        process_factory=execute,
        encoding=None,
    )
    try:
        async with asyncssh.connect("127.0.0.1", port=server.get_port(), known_hosts=None) as c:
            root = tmp_path / "installation"
            source = tmp_path / "source.tar.gz"
            content = b"runtime-seed" * 300_000
            source.write_bytes(content)
            await upload_file(c, source, str(root), "release.tar.gz")
            assert (root / "release.tar.gz").read_bytes() == content
            await upload_bytes(c, b"new-pointer\n", str(root), "current")
            assert (root / "current").read_bytes() == b"new-pointer\n"
    finally:
        server.close()
        await server.wait_closed()
