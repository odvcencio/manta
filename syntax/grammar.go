package syntax

import (
	"fmt"
	"sync"

	gotreesitter "github.com/odvcencio/gotreesitter"
	"github.com/odvcencio/gotreesitter/grammargen"
)

var (
	mantaLangOnce sync.Once
	mantaLang     *gotreesitter.Language
	mantaLangErr  error
)

// MantaGrammar returns the tree-sitter grammar for the Manta source language.
func MantaGrammar() *grammargen.Grammar {
	g := grammargen.NewGrammar("manta")

	g.Define("source_file", grammargen.Repeat(grammargen.Sym("_declaration")))

	g.Define("_declaration", grammargen.Choice(
		grammargen.Sym("param_declaration"),
		grammargen.Sym("kernel_declaration"),
		grammargen.Sym("pipeline_declaration"),
	))

	g.Define("param_declaration", grammargen.Seq(
		grammargen.Str("param"),
		grammargen.Field("name", grammargen.Sym("identifier")),
		grammargen.Str(":"),
		grammargen.Field("type", grammargen.Sym("type_reference")),
		grammargen.Field("binding", grammargen.Sym("weight_annotation")),
		grammargen.Optional(grammargen.Field("trainable", grammargen.Sym("trainable_annotation"))),
	))

	g.Define("kernel_declaration", grammargen.Seq(
		grammargen.Str("kernel"),
		grammargen.Field("name", grammargen.Sym("identifier")),
		grammargen.Field("parameters", grammargen.Sym("parameter_list")),
		grammargen.Sym("arrow"),
		grammargen.Field("result", grammargen.Sym("type_reference")),
		grammargen.Field("body", grammargen.Sym("block")),
	))

	g.Define("pipeline_declaration", grammargen.Seq(
		grammargen.Str("pipeline"),
		grammargen.Field("name", grammargen.Sym("identifier")),
		grammargen.Field("parameters", grammargen.Sym("parameter_list")),
		grammargen.Sym("arrow"),
		grammargen.Field("result", grammargen.Choice(
			grammargen.Sym("type_reference"),
			grammargen.Sym("result_list"),
		)),
		grammargen.Field("body", grammargen.Sym("block")),
	))

	g.Define("parameter_list", grammargen.Seq(
		grammargen.Str("("),
		grammargen.Optional(grammargen.CommaSep1(grammargen.Sym("field"))),
		grammargen.Str(")"),
	))

	g.Define("result_list", grammargen.Seq(
		grammargen.Str("("),
		grammargen.CommaSep1(grammargen.Sym("field")),
		grammargen.Str(")"),
	))

	g.Define("field", grammargen.Seq(
		grammargen.Field("name", grammargen.Sym("identifier")),
		grammargen.Str(":"),
		grammargen.Field("type", grammargen.Sym("type_reference")),
	))

	g.Define("type_reference", grammargen.Seq(
		grammargen.Field("name", grammargen.Sym("identifier")),
		grammargen.Optional(grammargen.Field("shape", grammargen.Sym("shape"))),
	))

	g.Define("shape", grammargen.Seq(
		grammargen.Str("["),
		grammargen.CommaSep1(grammargen.Sym("dimension")),
		grammargen.Str("]"),
	))

	g.Define("dimension", grammargen.Choice(
		grammargen.Sym("identifier"),
		grammargen.Sym("number"),
	))

	g.Define("weight_annotation", grammargen.Seq(
		grammargen.Str("@"),
		grammargen.Str("weight"),
		grammargen.Str("("),
		grammargen.Field("binding", grammargen.Sym("string_literal")),
		grammargen.Str(")"),
	))

	g.Define("trainable_annotation", grammargen.Seq(
		grammargen.Str("@"),
		grammargen.Str("trainable"),
	))

	g.Define("block", grammargen.Seq(
		grammargen.Str("{"),
		grammargen.Repeat(grammargen.Sym("_statement")),
		grammargen.Str("}"),
	))

	g.Define("_statement", grammargen.Choice(
		grammargen.Sym("let_statement"),
		grammargen.Sym("return_statement"),
		grammargen.Sym("expression_statement"),
	))

	g.Define("let_statement", grammargen.Seq(
		grammargen.Str("let"),
		grammargen.Field("name", grammargen.Sym("identifier")),
		grammargen.Str("="),
		grammargen.Field("value", grammargen.Sym("_expression")),
	))

	g.Define("return_statement", grammargen.Seq(
		grammargen.Str("return"),
		grammargen.Field("values", grammargen.Sym("expression_list")),
	))

	g.Define("expression_statement", grammargen.Field("expression", grammargen.Sym("_expression")))

	g.Define("expression_list", grammargen.CommaSep1(grammargen.Sym("_expression")))

	g.Define("_expression", grammargen.Choice(
		grammargen.Sym("binary_expression"),
		grammargen.Sym("unary_expression"),
		grammargen.Sym("call_expression"),
		grammargen.Sym("intrinsic_call_expression"),
		grammargen.Sym("parenthesized_expression"),
		grammargen.Sym("identifier"),
		grammargen.Sym("number"),
		grammargen.Sym("string_literal"),
	))

	g.Define("binary_expression", grammargen.Choice(
		grammargen.PrecLeft(1, grammargen.Seq(
			grammargen.Field("left", grammargen.Sym("_expression")),
			grammargen.Field("operator", grammargen.Choice(grammargen.Str("+"), grammargen.Str("-"))),
			grammargen.Field("right", grammargen.Sym("_expression")),
		)),
		grammargen.PrecLeft(2, grammargen.Seq(
			grammargen.Field("left", grammargen.Sym("_expression")),
			grammargen.Field("operator", grammargen.Choice(grammargen.Str("*"), grammargen.Str("/"))),
			grammargen.Field("right", grammargen.Sym("_expression")),
		)),
	))

	g.Define("unary_expression", grammargen.Prec(3, grammargen.Seq(
		grammargen.Field("operator", grammargen.Str("-")),
		grammargen.Field("operand", grammargen.Sym("_expression")),
	)))

	g.Define("parenthesized_expression", grammargen.Seq(
		grammargen.Str("("),
		grammargen.Field("expression", grammargen.Sym("_expression")),
		grammargen.Str(")"),
	))

	g.Define("call_expression", grammargen.Seq(
		grammargen.Field("callee", grammargen.Sym("identifier")),
		grammargen.Field("arguments", grammargen.Sym("argument_list")),
	))

	g.Define("intrinsic_call_expression", grammargen.Seq(
		grammargen.Str("@"),
		grammargen.Field("callee", grammargen.Sym("identifier")),
		grammargen.Field("arguments", grammargen.Sym("argument_list")),
	))

	g.Define("argument_list", grammargen.Seq(
		grammargen.Str("("),
		grammargen.Optional(grammargen.CommaSep1(grammargen.Sym("_expression"))),
		grammargen.Str(")"),
	))

	g.Define("arrow", grammargen.Token(grammargen.Seq(grammargen.Str("-"), grammargen.Str(">"))))
	g.Define("identifier", grammargen.Pat(`[A-Za-z_][A-Za-z0-9_]*`))
	g.Define("number", grammargen.Token(grammargen.Pat(`[0-9]+(\.[0-9]+)?`)))
	g.Define("string_literal", grammargen.Token(grammargen.Seq(
		grammargen.Str(`"`),
		grammargen.Pat(`([^"\\]|\\.)*`),
		grammargen.Str(`"`),
	)))
	g.Define("comment", grammargen.Token(grammargen.Pat(`//[^\n\x00]*`)))

	g.SetExtras(
		grammargen.Pat(`[ \t\r\n]+`),
		grammargen.Sym("comment"),
	)
	g.SetWord("identifier")

	g.Test("tiny embed", `
param token_embedding: f16[V, D] @weight("weights/token_embedding")

pipeline embed(tokens: i32[T]) -> f16[T, D] {
    let hidden = gather(token_embedding, tokens)
    return normalize(hidden)
}
`, "")

	return g
}

// Language returns the cached generated Manta tree-sitter language.
func Language() (*gotreesitter.Language, error) {
	mantaLangOnce.Do(func() {
		mantaLang, mantaLangErr = grammargen.GenerateLanguage(MantaGrammar())
	})
	return mantaLang, mantaLangErr
}

// ParseTree parses Manta source and returns the tree-sitter tree and language.
func ParseTree(src []byte) (*gotreesitter.Tree, *gotreesitter.Language, error) {
	lang, err := Language()
	if err != nil {
		return nil, nil, fmt.Errorf("generate Manta language: %w", err)
	}
	parser := gotreesitter.NewParser(lang)
	tree, err := parser.Parse(src)
	if err != nil {
		return nil, nil, fmt.Errorf("parse Manta source: %w", err)
	}
	return tree, lang, nil
}
