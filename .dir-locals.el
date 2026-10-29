;; ruff lints this project: lsp-mode starts its `ruff server' add-on client because direnv puts
;; ruff on PATH. pylsp stays for navigation and completion with its own linters off, so issues are
;; not reported twice.
((python-base-mode . ((lsp-pylsp-plugins-flake8-enabled . nil)
                      (lsp-pylsp-plugins-pycodestyle-enabled . nil)
                      (lsp-pylsp-plugins-pyflakes-enabled . nil)
                      (lsp-pylsp-plugins-pydocstyle-enabled . nil)
                      (lsp-pylsp-plugins-mccabe-enabled . nil))))
