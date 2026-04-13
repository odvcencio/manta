package mantaruntime

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"strings"
	"unicode"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	mll "github.com/odvcencio/mll"
)

const TokenizerFileVersion = "manta/tokenizer/v0alpha1"

var tagXTOK = [4]byte{'X', 'T', 'O', 'K'}

type TokenizerMerge struct {
	Left  string `json:"left"`
	Right string `json:"right"`
}

// TokenizerFile is a lightweight Manta text-tokenizer bundle for training-time ingestion.
type TokenizerFile struct {
	Version      string           `json:"version"`
	Tokens       []string         `json:"tokens"`
	Merges       []TokenizerMerge `json:"merges,omitempty"`
	PadToken     string           `json:"pad_token,omitempty"`
	UnknownToken string           `json:"unknown_token,omitempty"`
	BOSToken     string           `json:"bos_token,omitempty"`
	EOSToken     string           `json:"eos_token,omitempty"`
}

func DefaultTokenizerPath(artifactPath string) string {
	return defaultManifestPath(artifactPath, ".tokenizer.mll")
}

func (f TokenizerFile) Validate() error {
	if f.Version == "" {
		return fmt.Errorf("tokenizer version is required")
	}
	if f.Version != TokenizerFileVersion {
		return fmt.Errorf("tokenizer version %q is not supported, want %q", f.Version, TokenizerFileVersion)
	}
	if len(f.Tokens) == 0 {
		return fmt.Errorf("tokenizer tokens are empty")
	}
	seen := map[string]bool{}
	for i, tok := range f.Tokens {
		if tok == "" {
			return fmt.Errorf("tokenizer token %d is empty", i)
		}
		if seen[tok] {
			return fmt.Errorf("duplicate tokenizer token %q", tok)
		}
		seen[tok] = true
	}
	return nil
}

func (f TokenizerFile) WriteFile(path string) error {
	if err := f.Validate(); err != nil {
		return err
	}
	data, err := encodeTokenizerMLL(f)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func ReadTokenizerFile(path string) (TokenizerFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return TokenizerFile{}, err
	}
	if !mantaartifact.IsMLLBytes(data) {
		return TokenizerFile{}, fmt.Errorf("tokenizer %q is not an MLL file", path)
	}
	return decodeTokenizerMLL(data)
}

func encodeTokenizerMLL(file TokenizerFile) ([]byte, error) {
	strg := mll.NewStringTableBuilder()
	strg.Intern("")
	for _, token := range file.Tokens {
		strg.Intern(token)
	}
	for _, merge := range file.Merges {
		strg.Intern(merge.Left)
		strg.Intern(merge.Right)
	}
	strg.Intern(file.PadToken)
	strg.Intern(file.UnknownToken)
	strg.Intern(file.BOSToken)
	strg.Intern(file.EOSToken)

	head := mll.HeadSection{
		Name:        strg.Intern("manta-tokenizer"),
		Description: strg.Intern("Manta tokenizer"),
		Metadata: []mll.HeadMetadataEntry{
			headStringMeta(strg, "tokenizer_version", file.Version),
			headIntMeta(strg, "token_count", int64(len(file.Tokens))),
			headIntMeta(strg, "merge_count", int64(len(file.Merges))),
		},
	}

	var xbody bytes.Buffer
	writeU32 := func(v uint32) error {
		return binary.Write(&xbody, binary.LittleEndian, v)
	}
	if err := writeU32(uint32(len(file.Tokens))); err != nil {
		return nil, err
	}
	if err := writeU32(uint32(len(file.Merges))); err != nil {
		return nil, err
	}
	for _, special := range []string{file.PadToken, file.UnknownToken, file.BOSToken, file.EOSToken} {
		if err := writeU32(strg.Intern(special)); err != nil {
			return nil, err
		}
	}
	for _, token := range file.Tokens {
		if err := writeU32(strg.Intern(token)); err != nil {
			return nil, err
		}
	}
	for _, merge := range file.Merges {
		if err := writeU32(strg.Intern(merge.Left)); err != nil {
			return nil, err
		}
		if err := writeU32(strg.Intern(merge.Right)); err != nil {
			return nil, err
		}
	}

	var (
		dimsBuilder mll.DimsBuilder
		typeBuilder mll.TypeBuilder
		parmBuilder mll.ParmBuilder
		entrBuilder mll.EntrBuilder
		tnsrBuilder mll.TnsrBuilder
	)

	sections := make([]mll.SectionInput, 0, 9)
	if body, digestBody, err := encodeHeadSection(head, mll.ProfileSealed); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagHEAD, Body: body, DigestBody: digestBody, Flags: mll.SectionFlagRequired, SchemaVersion: 1})
	}
	if body, err := encodeSection(strg.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagSTRG, Body: body, Flags: mll.SectionFlagRequired, SchemaVersion: 1})
	}
	if body, err := encodeSection(dimsBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagDIMS, Body: body, Flags: mll.SectionFlagRequired, SchemaVersion: 1})
	}
	if body, err := encodeSection(typeBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagTYPE, Body: body, SchemaVersion: 1})
	}
	if body, err := encodeSection(parmBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagPARM, Body: body, Flags: mll.SectionFlagRequired, SchemaVersion: 1})
	}
	if body, err := encodeSection(entrBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagENTR, Body: body, Flags: mll.SectionFlagRequired, SchemaVersion: 1})
	}
	if body, err := encodeSection(tnsrBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagTNSR, Body: body, Flags: mll.SectionFlagRequired | mll.SectionFlagAligned, SchemaVersion: 1})
	}
	if body, err := encodeSection((mll.SchmSection{}).Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagSCHM, Body: body, SchemaVersion: 1})
	}
	sections = append(sections, mll.SectionInput{
		Tag:           tagXTOK,
		Body:          xbody.Bytes(),
		Flags:         mll.SectionFlagSkippable | mll.SectionFlagSchemaless,
		SchemaVersion: 1,
	})
	return mll.WriteToBytes(mll.ProfileSealed, mll.V1_0, sections)
}

func decodeTokenizerMLL(data []byte) (TokenizerFile, error) {
	reader, err := mll.ReadBytes(data, mll.WithDigestVerification())
	if err != nil {
		return TokenizerFile{}, err
	}
	if reader.Profile() != mll.ProfileSealed {
		return TokenizerFile{}, fmt.Errorf("tokenizer profile = %d, want %d", reader.Profile(), mll.ProfileSealed)
	}
	strgBody, ok := reader.Section(mll.TagSTRG)
	if !ok {
		return TokenizerFile{}, fmt.Errorf("tokenizer missing STRG section")
	}
	strg, err := mll.ReadStringTable(strgBody)
	if err != nil {
		return TokenizerFile{}, err
	}
	body, ok := reader.Section(tagXTOK)
	if !ok {
		return TokenizerFile{}, fmt.Errorf("tokenizer missing XTOK section")
	}
	r := bytes.NewReader(body)
	readU32 := func() (uint32, error) {
		var v uint32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	}
	tokenCount, err := readU32()
	if err != nil {
		return TokenizerFile{}, err
	}
	mergeCount, err := readU32()
	if err != nil {
		return TokenizerFile{}, err
	}
	specials := make([]uint32, 4)
	for i := range specials {
		specials[i], err = readU32()
		if err != nil {
			return TokenizerFile{}, err
		}
	}
	file := TokenizerFile{Version: TokenizerFileVersion}
	file.PadToken = strg.At(specials[0])
	file.UnknownToken = strg.At(specials[1])
	file.BOSToken = strg.At(specials[2])
	file.EOSToken = strg.At(specials[3])
	file.Tokens = make([]string, int(tokenCount))
	for i := range file.Tokens {
		idx, err := readU32()
		if err != nil {
			return TokenizerFile{}, err
		}
		file.Tokens[i] = strg.At(idx)
	}
	file.Merges = make([]TokenizerMerge, int(mergeCount))
	for i := range file.Merges {
		leftIdx, err := readU32()
		if err != nil {
			return TokenizerFile{}, err
		}
		rightIdx, err := readU32()
		if err != nil {
			return TokenizerFile{}, err
		}
		file.Merges[i] = TokenizerMerge{Left: strg.At(leftIdx), Right: strg.At(rightIdx)}
	}
	if err := file.Validate(); err != nil {
		return TokenizerFile{}, err
	}
	return file, nil
}

type BPETokenizer struct {
	tokenToID  map[string]int32
	merges     []TokenizerMerge
	mergeRanks map[string]map[string]rankedBPEMerge
	unknownID  int32
	bosID      int32
	eosID      int32
	hasBOS     bool
	hasEOS     bool
	maxLen     int
}

type rankedBPEMerge struct {
	rank  int
	token string
}

func NewBPETokenizer(file TokenizerFile, manifest TokenizerManifest) (*BPETokenizer, error) {
	if err := file.Validate(); err != nil {
		return nil, err
	}
	if manifest.VocabSize > 0 && len(file.Tokens) > manifest.VocabSize {
		return nil, fmt.Errorf("tokenizer size %d exceeds manifest vocab_size %d", len(file.Tokens), manifest.VocabSize)
	}
	tokenToID := make(map[string]int32, len(file.Tokens))
	for i, tok := range file.Tokens {
		tokenToID[tok] = int32(i)
	}
	mergeRanks := make(map[string]map[string]rankedBPEMerge, len(file.Merges))
	for rank, merge := range file.Merges {
		rights := mergeRanks[merge.Left]
		if rights == nil {
			rights = map[string]rankedBPEMerge{}
			mergeRanks[merge.Left] = rights
		}
		if _, exists := rights[merge.Right]; exists {
			continue
		}
		rights[merge.Right] = rankedBPEMerge{
			rank:  rank,
			token: merge.Left + merge.Right,
		}
	}
	padTok := file.PadToken
	if padTok == "" {
		padTok = "[PAD]"
	}
	_ = padTok
	unkTok := file.UnknownToken
	if unkTok == "" {
		unkTok = "[UNK]"
	}
	bosTok := file.BOSToken
	if bosTok == "" {
		bosTok = "[CLS]"
	}
	eosTok := file.EOSToken
	if eosTok == "" {
		eosTok = "[SEP]"
	}
	unknownID := int32(-1)
	if id, ok := tokenToID[unkTok]; ok {
		unknownID = id
	}
	out := &BPETokenizer{
		tokenToID:  tokenToID,
		merges:     append([]TokenizerMerge(nil), file.Merges...),
		mergeRanks: mergeRanks,
		unknownID:  unknownID,
		maxLen:     manifest.MaxSequence,
	}
	if id, ok := tokenToID[bosTok]; ok {
		out.bosID = id
		out.hasBOS = true
	}
	if id, ok := tokenToID[eosTok]; ok {
		out.eosID = id
		out.hasEOS = true
	}
	return out, nil
}

func (t *BPETokenizer) Encode(text string) ([]int32, []int32, error) {
	if t == nil {
		return nil, nil, fmt.Errorf("nil tokenizer")
	}
	toks := bpeMergeRanked(splitChars(normalizeText(text)), t.mergeRanks)
	ids := make([]int32, 0, len(toks)+2)
	if t.hasBOS {
		ids = append(ids, t.bosID)
	}
	for _, tok := range toks {
		if id, ok := t.tokenToID[tok]; ok {
			ids = append(ids, id)
			continue
		}
		if t.unknownID < 0 {
			return nil, nil, fmt.Errorf("token %q is not in tokenizer vocabulary and no unknown token is configured", tok)
		}
		ids = append(ids, t.unknownID)
	}
	if t.hasEOS {
		ids = append(ids, t.eosID)
	}
	if t.maxLen > 0 && len(ids) > t.maxLen {
		ids = append([]int32(nil), ids[:t.maxLen]...)
		if t.hasEOS {
			ids[len(ids)-1] = t.eosID
		}
	}
	if len(ids) == 0 {
		return nil, nil, fmt.Errorf("tokenized text is empty")
	}
	mask := make([]int32, len(ids))
	for i := range mask {
		mask[i] = 1
	}
	return ids, mask, nil
}

func normalizeText(text string) string {
	var b strings.Builder
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(unicode.ToLower(r))
		} else {
			b.WriteRune(' ')
		}
	}
	return strings.TrimSpace(b.String())
}

func splitChars(text string) []string {
	parts := strings.Fields(text)
	chars := make([]string, 0, len(text))
	for i, word := range parts {
		if i > 0 {
			chars = append(chars, " ")
		}
		for _, r := range word {
			chars = append(chars, string(r))
		}
	}
	return chars
}

func bpeMerge(tokens []string, merges []TokenizerMerge) []string {
	return bpeMergeRanked(tokens, mergeRankLookup(merges))
}

func mergeRankLookup(merges []TokenizerMerge) map[string]map[string]rankedBPEMerge {
	out := make(map[string]map[string]rankedBPEMerge, len(merges))
	for rank, merge := range merges {
		rights := out[merge.Left]
		if rights == nil {
			rights = map[string]rankedBPEMerge{}
			out[merge.Left] = rights
		}
		if _, exists := rights[merge.Right]; exists {
			continue
		}
		rights[merge.Right] = rankedBPEMerge{
			rank:  rank,
			token: merge.Left + merge.Right,
		}
	}
	return out
}

func bpeMergeRanked(tokens []string, merges map[string]map[string]rankedBPEMerge) []string {
	if len(tokens) < 2 || len(merges) == 0 {
		return tokens
	}
	for {
		bestRank := int(^uint(0) >> 1)
		var bestLeft, bestRight, bestToken string
		found := false
		for i := 0; i < len(tokens)-1; i++ {
			rights := merges[tokens[i]]
			if rights == nil {
				continue
			}
			merge, ok := rights[tokens[i+1]]
			if !ok || (found && merge.rank >= bestRank) {
				continue
			}
			bestRank = merge.rank
			bestLeft = tokens[i]
			bestRight = tokens[i+1]
			bestToken = merge.token
			found = true
		}
		if !found {
			return tokens
		}
		tokens = applyRankedMerge(tokens, bestLeft, bestRight, bestToken)
	}
}

func applyRankedMerge(tokens []string, left, right, merged string) []string {
	if len(tokens) < 2 {
		return tokens
	}
	write := 0
	for read := 0; read < len(tokens); {
		if read < len(tokens)-1 && tokens[read] == left && tokens[read+1] == right {
			tokens[write] = merged
			write++
			read += 2
			continue
		}
		tokens[write] = tokens[read]
		write++
		read++
	}
	return tokens[:write]
}

func applyMerge(tokens []string, left, right string) []string {
	if len(tokens) < 2 {
		return tokens
	}
	out := make([]string, 0, len(tokens))
	for i := 0; i < len(tokens); {
		if i < len(tokens)-1 && tokens[i] == left && tokens[i+1] == right {
			out = append(out, left+right)
			i += 2
			continue
		}
		out = append(out, tokens[i])
		i++
	}
	return out
}
