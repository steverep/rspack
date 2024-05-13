# rspack-ast-viewer

Print AST generated by SWC.

## Options

- -k, --keep-span: keep span information from the AST (default: false)

## Example

Input:

```bash
echo "var a = 123" | cargo run -p rspack_ast_viewer
```

Output:

```
Module(
    Module {
        body: [
            Stmt(
                Decl(
                    Var(
                        VarDecl {
                            kind: "var",
                            declare: false,
                            decls: [
                                VarDeclarator {
                                    name: Ident(
                                        BindingIdent {
                                            id: Ident {
                                                sym: Atom('abc' type=inline),
                                                optional: false,
                                            },
                                            type_ann: None,
                                        },
                                    ),
                                    init: Some(
                                        Lit(
                                            Num(
                                                Number {
                                                    value: 123.0,
                                                    raw: Some(
                                                        "123",
                                                    ),
                                                },
                                            ),
                                        ),
                                    ),
                                    definite: false,
                                },
                            ],
                        },
                    ),
                ),
            ),
        ],
        shebang: None,
    },
)
```