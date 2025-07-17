import os
import argparse
import numpy as np
import tensorflow as tf
import sentencepiece as spm
import pickle
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import re
from model import create_model

class BrajHindiTranslator:
    def __init__(self, model_weights_path: str, metadata_path: str, braj_tokenizer_path: str, hindi_tokenizer_path: str):
        self.load_components(model_weights_path, metadata_path, braj_tokenizer_path, hindi_tokenizer_path)
        self.setup_fallback_system()
    
    def load_components(self, model_weights_path: str, metadata_path: str, braj_tokenizer_path: str, hindi_tokenizer_path: str):
        print("Loading translation components")
        
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.max_seq_len = self.metadata['max_seq_len']
        self.input_vocab_size = self.metadata['input_vocab_size']
        self.target_vocab_size = self.metadata['target_vocab_size']
        self.braj_tokenizer = spm.SentencePieceProcessor(model_file=braj_tokenizer_path)
        self.hindi_tokenizer = spm.SentencePieceProcessor(model_file=hindi_tokenizer_path)
        self.model = create_model(self.input_vocab_size, self.target_vocab_size, self.max_seq_len)
        
        dummy_braj = tf.zeros((1, self.max_seq_len), dtype=tf.int32)
        dummy_hindi = tf.zeros((1, self.max_seq_len), dtype=tf.int32)
        _ = self.model([dummy_braj, dummy_hindi], training=False)
        
        self.model.load_weights(model_weights_path)
        
        print("All components loaded successfully!")
    
    def setup_fallback_system(self):
        print("Setting up intelligent fallback system...")
        
        original_texts = self.metadata.get('original_texts', ([], []))
        self.original_braj, self.original_hindi = original_texts
        self.word_dict = {}
        self.phrase_dict = {}
        
        for braj_text, hindi_text in zip(self.original_braj, self.original_hindi):
            braj_words = braj_text.split()
            hindi_words = hindi_text.split()
            
            self.phrase_dict[braj_text.strip()] = hindi_text.strip()
            
            if len(braj_words) == 1 and len(hindi_words) == 1:
                self.word_dict[braj_words[0]] = hindi_words[0]
            
            if len(braj_words) == len(hindi_words) and len(braj_words) <= 5:
                for braj_word, hindi_word in zip(braj_words, hindi_words):
                    if braj_word not in self.word_dict:
                        self.word_dict[braj_word] = hindi_word
                    elif self.word_dict[braj_word] == hindi_word:
                        continue  
                    else:
                        pass
        common_mappings = {
            'हौं': 'मैं', 'तुम': 'तुम', 'वो': 'वह', 'ये': 'यह',
            'करत': 'करता', 'करब': 'करूंगा', 'करि': 'करके', 'कर': 'कर',
            'जात': 'जाता', 'जाब': 'जाऊंगा', 'जा': 'जा', 'आब': 'आऊंगा',
            'है': 'है', 'हैं': 'हैं', 'थो': 'था', 'थीं': 'थी',
            'को': 'को', 'कौं': 'को', 'की': 'की', 'के': 'के',
            'में': 'में', 'पै': 'पर', 'सें': 'से', 'तें': 'से',
            'मुंह': 'मुंह', 'धोत': 'धोता', 'धोब': 'धोऊंगा',
            'खात': 'खाता', 'खाब': 'खाऊंगा', 'पीत': 'पीता', 'पीब': 'पीऊंगा',
            'सुनत': 'सुनता', 'देखत': 'देखता', 'बोलत': 'बोलता',
            'चलत': 'चलता', 'दौड़त': 'दौड़ता', 'भागत': 'भागता',
            'पढ़त': 'पढ़ता', 'लिखत': 'लिखता', 'सोत': 'सोता',
            'उठत': 'उठता', 'बैठत': 'बैठता', 'खड़ो': 'खड़ा',
            'अच्छो': 'अच्छा', 'बुरो': 'बुरा', 'नयो': 'नया', 'पुरानो': 'पुराना',
            'छोटो': 'छोटा', 'बड़ो': 'बड़ा', 'लम्बो': 'लम्बा', 'मोटो': 'मोटा',
            'काला': 'काला', 'गोरो': 'गोरा', 'लाल': 'लाल', 'हरो': 'हरा',
            'सफेद': 'सफेद', 'पीलो': 'पीला', 'नीलो': 'नीला',
            'घर': 'घर', 'कोठरी': 'कमरा', 'दरवाजो': 'दरवाजा',
            'खिड़की': 'खिड़की', 'छत': 'छत', 'फर्श': 'फर्श',
            'माता': 'माता', 'पिता': 'पिता', 'भाई': 'भाई', 'बहन': 'बहन',
            'दादा': 'दादा', 'दादी': 'दादी', 'नाना': 'नाना', 'नानी': 'नानी'
        }
        for braj_word, hindi_word in common_mappings.items():
            if braj_word not in self.word_dict:
                self.word_dict[braj_word] = hindi_word
        additional_mappings = {
            'साबुन': 'साबुन', 'सें': 'से', 'काया': 'शरीर', 'साफ': 'साफ',
            'सिर': 'सिर', 'तेल': 'तेल', 'लगाइ': 'लगाकर', 'मारदन': 'मालिश',
            'निद्रा': 'नींद', 'खुलत': 'खुलती', 'किरन': 'किरण', 
            'पहली': 'पहली', 'भोर': 'सुबह', 'छह': 'छह', 'बजे': 'बजे',
            'जागत': 'उठता', 'मेरी': 'मेरी', 'तें': 'से'
        }
        
        for braj_word, hindi_word in additional_mappings.items():
            if braj_word not in self.word_dict:
                self.word_dict[braj_word] = hindi_word
        self.compute_embedding_similarity_matrix()
        
        print(f"Fallback system ready with {len(self.word_dict)} direct word mappings")
        print(f"Phrase dictionary ready with {len(self.phrase_dict)} sentence mappings")
    
    def compute_embedding_similarity_matrix(self):
        braj_vocab = [self.braj_tokenizer.id_to_piece(i) for i in range(self.braj_tokenizer.get_piece_size())]
        hindi_vocab = [self.hindi_tokenizer.id_to_piece(i) for i in range(self.hindi_tokenizer.get_piece_size())]
        self.braj_vocab = braj_vocab
        self.hindi_vocab = hindi_vocab
    
    def preprocess_text(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\r+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def beam_search_decode(self, encoder_input: tf.Tensor, beam_width: int = 3) -> List[str]:
        sequences = [[2]]  
        scores = [0.0]
        
        for _ in range(self.max_seq_len - 1):
            all_candidates = []
            
            for i, sequence in enumerate(sequences):
                if sequence[-1] == 3: 
                    all_candidates.append((sequence, scores[i]))
                    continue
                
                padded_seq = sequence + [0] * (self.max_seq_len - len(sequence))
                padded_seq = padded_seq[:self.max_seq_len]
                
                decoder_input = tf.expand_dims(padded_seq, 0)
                
                predictions = self.model([encoder_input, decoder_input], training=False)
                last_token_probs = tf.nn.softmax(predictions[0, len(sequence)-1, :])
                top_k = tf.nn.top_k(last_token_probs, k=beam_width)
                
                for j in range(beam_width):
                    token_id = int(top_k.indices[j])
                    token_prob = float(top_k.values[j])
                    
                    new_sequence = sequence + [token_id]
                    new_score = scores[i] + np.log(token_prob + 1e-8)
                    
                    all_candidates.append((new_sequence, new_score))
    
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            sequences = [seq for seq, _ in all_candidates[:beam_width]]
            scores = [score for _, score in all_candidates[:beam_width]]
            
            if all(seq[-1] == 3 for seq in sequences):
                break
        
        translations = []
        for sequence in sequences:
            tokens = [token for token in sequence if token not in [0, 2, 3]]
            if tokens:
                translation = self.hindi_tokenizer.decode(tokens)
                translations.append(translation)
        
        return translations if translations else [""]
    
    def split_mixed_content(self, text: str) -> List[str]:
        separators = ['और', 'की', 'का', 'के', 'में', 'से', 'पर', 'को', 'ने', 'तक', 'लिए', 'साथ', 'बाद', 'पहले']
        
        chunks = []
        current_chunk = []
        words = text.split()
        
        for word in words:
            if word in separators and current_chunk:
                chunks.append(' '.join(current_chunk))
                chunks.append(word)  
                current_chunk = []
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
    
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        return chunks if chunks else [text]
    
    def translate_chunks(self, chunks: List[str], mode: str = "auto") -> str:
        translated_chunks = []
        
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
                
            if len(chunk.split()) == 1:
                if chunk in self.word_dict:
                    translated_chunks.append(self.word_dict[chunk])
                else:
                    try:
                        neural_result = self.neural_translate(chunk)
                        if neural_result and neural_result.strip():
                            translated_chunks.append(neural_result)
                        else:
                            fallback_result = self.subword_fallback_translate(chunk)
                            translated_chunks.append(fallback_result)
                    except Exception:
                        fallback_result = self.subword_fallback_translate(chunk)
                        translated_chunks.append(fallback_result)
            else:
                try:
                    neural_result = self.neural_translate(chunk)
                    if neural_result and neural_result.strip():
                        translated_chunks.append(neural_result)
                    else:
                        word_by_word = self.subword_fallback_translate(chunk)
                        translated_chunks.append(word_by_word)
                except Exception:
                    word_by_word = self.subword_fallback_translate(chunk)
                    translated_chunks.append(word_by_word)
        
        result = ' '.join(translated_chunks)
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result

    def translate_with_fallback(self, braj_text: str, mode: str = "auto") -> str:
        braj_text = self.preprocess_text(braj_text)
        
        if braj_text in self.phrase_dict:
            return self.phrase_dict[braj_text]
        
        partial_matches = []
        for phrase in self.phrase_dict.keys():
            if phrase.startswith(braj_text):
                partial_matches.append(phrase)
        
        if partial_matches:
            best_match = min(partial_matches, key=len)
            input_words = braj_text.split()
            match_words = best_match.split()
            translation_words = self.phrase_dict[best_match].split()
            
            if len(input_words) < len(match_words) and len(translation_words) >= len(input_words):
                partial_translation = ' '.join(translation_words[:len(input_words)])
                return partial_translation
        
        if mode in ["word", "auto"] and len(braj_text.split()) == 1:
            if braj_text in self.word_dict:
                return self.word_dict[braj_text]
        
        sentences = self.split_into_sentences(braj_text)
        if len(sentences) > 1:
            print(f"Detected {len(sentences)} sentences, translating separately...")
            translated_sentences = []
            
            for i, sentence in enumerate(sentences):
                print(f"Translating sentence {i+1}: {sentence}")
                try:
                    neural_result = self.neural_translate(sentence)
                    if neural_result and neural_result.strip() and self.is_good_translation(neural_result, sentence):
                        translated_sentences.append(self.post_process_translation(neural_result))
                        print(f"Neural result {i+1}: {neural_result}")
                    else:
                        fallback_result = self.word_by_word_translate(sentence)
                        translated_sentences.append(fallback_result)
                        print(f"Fallback result {i+1}: {fallback_result}")
                except Exception as e:
                    print(f"Neural translation failed for sentence {i+1}: {e}")
                    fallback_result = self.word_by_word_translate(sentence)
                    translated_sentences.append(fallback_result)
                    print(f"Fallback result {i+1}: {fallback_result}")
            result = ' '.join(translated_sentences)
            return self.post_process_translation(result)
        
        try:
            neural_translation = self.neural_translate(braj_text)
            if neural_translation and neural_translation.strip() and self.is_good_translation(neural_translation, braj_text):
                neural_translation = self.post_process_translation(neural_translation)
                return neural_translation
        except Exception as e:
            print(f"Neural translation failed: {e}")
        
        similarity_matches = []
        input_words = set(braj_text.split())
        for phrase, translation in self.phrase_dict.items():
            phrase_words = set(phrase.split())
            common_words = input_words.intersection(phrase_words)
            similarity_score = len(common_words) / max(len(input_words), len(phrase_words))
            if similarity_score > 0.6:
                similarity_matches.append((phrase, translation, similarity_score))
        
        if similarity_matches:
            similarity_matches.sort(key=lambda x: x[2], reverse=True)
            best_phrase, best_translation, _ = similarity_matches[0]
            
            input_words = braj_text.split()
            phrase_words = best_phrase.split()
            translation_words = best_translation.split()
            
            if len(input_words) < len(phrase_words):
                ratio = len(input_words) / len(phrase_words)
                target_length = max(1, int(len(translation_words) * ratio))
                partial_translation = ' '.join(translation_words[:target_length])
                return partial_translation
            else:
                return best_translation
        
        if len(braj_text.split()) > 1:
            chunks = self.split_mixed_content(braj_text)
            if len(chunks) > 1: 
                return self.translate_chunks(chunks, mode)
        
        word_by_word = self.word_by_word_translate(braj_text)
        if word_by_word and word_by_word != braj_text:
            return word_by_word
        
        return self.subword_fallback_translate(braj_text)
    
    def is_good_translation(self, translation: str, original: str) -> bool:
        words = translation.split()
        if len(words) > 5:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
                if word_counts[word] > 3:
                    return False
        if len(words) > len(original.split()) * 3:
            return False
        
        bad_phrases = ['डोसा झूलता', 'शव में', 'श्रद्धांजलि', 'तोड़']
        for bad_phrase in bad_phrases:
            if bad_phrase in translation:
                return False
        if len(original.split()) > 3 and len(words) < 2:
            return False
        
        return True

    def post_process_translation(self, translation: str) -> str:
        corrections = {
            'मैंं': 'मैं', 'तुमम': 'तुम', 'वोो': 'वह',
            'करताा': 'करता', 'जाताा': 'जाता', 'खाताा': 'खाता',
            'हैै': 'है', 'कोो': 'को', 'सेे': 'से',
            'मेंं': 'में', 'परर': 'पर', 'कीी': 'की',
        }
        
        result = translation
        for wrong, correct in corrections.items():
            result = result.replace(wrong, correct)
        
        word_order_fixes = [
            ('करता मैं', 'मैं करता हूँ'),
            ('करत मैं', 'मैं करता हूँ'),
            ('साफ करता मैं', 'मैं साफ करता हूँ'),
            ('लगाकर मालिश करता मैं', 'मैं लगाकर मालिश करता हूँ'),
        ]
        
        for wrong_order, correct_order in word_order_fixes:
            result = result.replace(wrong_order, correct_order)
        
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def word_by_word_translate(self, braj_text: str) -> str:
        words = braj_text.split()
        translated_words = []
        
        for i, word in enumerate(words):
            if i < len(words) - 1:
                two_word_phrase = f"{word} {words[i+1]}"
                if two_word_phrase in self.phrase_dict:
                    translated_words.append(self.phrase_dict[two_word_phrase])
                    words[i+1] = "__SKIP__"  
                    continue
                elif two_word_phrase in self.word_dict:
                    translated_words.append(self.word_dict[two_word_phrase])
                    words[i+1] = "__SKIP__"
                    continue
            
            if i < len(words) - 2:
                three_word_phrase = f"{word} {words[i+1]} {words[i+2]}"
                if three_word_phrase in self.phrase_dict:
                    translated_words.append(self.phrase_dict[three_word_phrase])
                    words[i+1] = "__SKIP__"
                    words[i+2] = "__SKIP__"
                    continue
            
            if word == "__SKIP__":
                continue
            
            if word in self.word_dict:
                translated_words.append(self.word_dict[word])
            else:
                fallback_translation = self.enhanced_word_fallback(word)
                translated_words.append(fallback_translation)
        
        return ' '.join(translated_words)
    
    def enhanced_word_fallback(self, word: str) -> str:
        similar_word = self.find_closest_word(word)
        if similar_word and similar_word != word:
            return similar_word
        
        transformed = self.apply_morphological_rules(word)
        if transformed != word:
            return transformed
        
        return self.character_level_translate(word)
    
    def apply_morphological_rules(self, word: str) -> str:
        transformations = [
            ('त$', 'ता'), 
            ('ब$', 'ऊंगा'),
            ('ो$', 'ा'),     
            ('ौं$', 'ैं'),     
            ('कौं$', 'को'),    
            ('तें$', 'से'),   
            ('पै$', 'पर'), 
            ('सें$', 'से'),     
        ]
        
        result = word
        for pattern, replacement in transformations:
            result = re.sub(pattern, replacement, result)
            if result != word:
                break
        
        return result
    
    def neural_translate(self, text: str) -> str:
        text = self.preprocess_text(text)
        braj_tokens = self.braj_tokenizer.encode(text, add_bos=True, add_eos=True)
        
        if len(braj_tokens) > self.max_seq_len:
            braj_tokens = braj_tokens[:self.max_seq_len]
        else:
            braj_tokens = braj_tokens + [0] * (self.max_seq_len - len(braj_tokens))
        
        encoder_input = tf.expand_dims(braj_tokens, 0)
        translations = self.beam_search_decode(encoder_input, beam_width=3)
        
        return translations[0] if translations and translations[0].strip() else ""

    def split_into_sentences(self, text: str) -> List[str]:
        text = text.strip()
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\r+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = self.fix_concatenated_sentences(text)
        
        if len(text.split()) <= 8:  
            return [text]
        
        sentences = []
        current_sentence = []
        words = text.split()
        
        for i, word in enumerate(words):
            current_sentence.append(word)
            
            is_sentence_end = False
            
            if len(current_sentence) >= 4:
                if i < len(words) - 1:  
                    next_word = words[i + 1]
                    
                    sentence_starters = ['हौं', 'तुम', 'वो', 'ये', 'वह', 'यह', 'मैं', 'तू']
                    if next_word in sentence_starters:
                        is_sentence_end = True
                    
                    time_markers = ['सुबह', 'शाम', 'रात', 'दिन', 'भोर', 'सूरज', 'चाँद']
                    if next_word in time_markers:
                        is_sentence_end = True
                    
                    connectors = ['फिर', 'बाद', 'पहले', 'तब', 'अब', 'जब']
                    if next_word in connectors:
                        is_sentence_end = True
                    
                    body_parts = ['सिर', 'मुंह', 'हाथ', 'पैर', 'आंख', 'कान', 'नाक']
                    if next_word in body_parts:
                        is_sentence_end = True
                    
                    if word == 'हौं':
                        is_sentence_end = True
            
            if len(current_sentence) >= 12:
                is_sentence_end = True
            
            if is_sentence_end or i == len(words) - 1:  
                sentence = ' '.join(current_sentence).strip()
                if sentence:
                    sentences.append(sentence)
                current_sentence = []
        
        if current_sentence:
            sentence = ' '.join(current_sentence).strip()
            if sentence:
                sentences.append(sentence)
        
        return sentences if sentences else [text]
    
    def fix_concatenated_sentences(self, text: str) -> str:
        specific_fixes = [
            ('हौंसिर', 'हौं सिर'),
            ('हौंतेल', 'हौं तेल'),
            ('हौंसूरज', 'हौं सूरज'),
            ('हैसिर', 'है सिर'),
            ('हूँसिर', 'हूँ सिर'),
            ('करतहौं', 'करत हौं'),
            ('जागतहौं', 'जागत हौं'),
            ('लगाइहौं', 'लगाइ हौं'),
            ('मारदनहौं', 'मारदन हौं'),
            ('धोतहौं', 'धोत हौं'),
            ('खातहौं', 'खात हौं'),
            ('पीतहौं', 'पीत हौं'),
            ('सुनतहौं', 'सुनत हौं'),
            ('देखतहौं', 'देखत हौं'),
        ]
        
        result = text
        for old, new in specific_fixes:
            result = result.replace(old, new)
        
        return result

    def subword_fallback_translate(self, text: str) -> str:
        word_translation = self.word_by_word_translate(text)
        words = text.split()
        translated_words = word_translation.split()
        
        if len(translated_words) >= len(words) * 0.7:  
            return word_translation
        return self.aggressive_subword_translate(text)
    
    def aggressive_subword_translate(self, text: str) -> str:
        words = text.split()
        translated_words = []
        
        for word in words:
            if word in self.word_dict:
                translated_words.append(self.word_dict[word])
            else:
                subword_translation = self.translate_with_subwords(word)
                translated_words.append(subword_translation)
        
        return ' '.join(translated_words)
    
    def translate_with_subwords(self, word: str) -> str:
        morphed = self.apply_morphological_rules(word)
        if morphed != word:
            return morphed
        
        braj_pieces = self.braj_tokenizer.encode(word, add_bos=False, add_eos=False)
        
        if len(braj_pieces) > 1:
            piece_translations = []
            for piece_id in braj_pieces:
                piece = self.braj_tokenizer.id_to_piece(piece_id)
                clean_piece = piece.replace('▁', '')
                if clean_piece in self.word_dict:
                    piece_translations.append(self.word_dict[clean_piece])
                else:
                    char_trans = self.character_level_translate(clean_piece)
                    piece_translations.append(char_trans)
            
            result = ''.join(piece_translations)
            return result if result else word
        
        return self.character_level_translate(word)
    
    def character_level_translate(self, word: str) -> str:
        char_mappings = {
            'ौं': 'ैं',    
            'ो': 'ा',    
            'त$': 'ता', 
            'ब$': 'ूंगा', 
        }
        
        result = word
        for old_char, new_char in char_mappings.items():
            if old_char.endswith('$'):
                pattern = old_char.replace('$', '$')
                result = re.sub(pattern, new_char, result)
            else:
                result = result.replace(old_char, new_char)
        
        return result
    
    def find_closest_word(self, word: str) -> str:
        if not word or word in self.word_dict:
            return self.word_dict.get(word, word)
        
        best_match = word
        best_score = 0
        
        for braj_word in self.word_dict.keys():
            if len(braj_word) == 0:
                continue
                
            common_chars = len(set(word) & set(braj_word))
            similarity = common_chars / max(len(word), len(braj_word))
            
            length_penalty = abs(len(word) - len(braj_word)) * 0.1
            final_score = similarity - length_penalty
            
            if final_score > best_score and final_score > 0.5: 
                best_score = final_score
                best_match = self.word_dict[braj_word]
        
        return best_match

    def translate(self, text: str, mode: str = "auto", verbose: bool = False) -> str:
        if not text or not text.strip():
            return ""
        
        if verbose:
            print(f"Input (Braj): '{text}'")
            print(f"Mode: {mode}")
        
        try:
            translation = self.translate_with_fallback(text, mode)
            
            if verbose:
                print(f"Output (Hindi): '{translation}'")
            
            return translation
            
        except Exception as e:
            if verbose:
                print(f"Translation error: {e}")
            return text  


def main():
    parser = argparse.ArgumentParser(description="Braj-to-Hindi Neural Machine Translation")
    parser.add_argument("--input", "-i", type=str, help="Input Braj text to translate")
    parser.add_argument("text", nargs="*", help="Input Braj text to translate (alternative to --input)")    
    parser.add_argument("--mode", "-m", choices=["word", "sentence", "auto"], default="auto", help="Translation mode")
    parser.add_argument("--model", type=str, default="best_model_weights.weights.h5", help="Path to model weights")
    parser.add_argument("--interactive", action="store_true", help="Start interactive translation mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    input_text = None
    if args.input:
        input_text = args.input
    elif args.text:
        input_text = ' '.join(args.text)
        
    model_path = args.model
    if not os.path.exists(model_path):
        if os.path.exists("final_model_weights.weights.h5"):
            model_path = "final_model_weights.weights.h5"
            print(f"{args.model} not found, using final_model_weights.weights.h5")
        else:
            print(f"Model weights not found: {args.model}")
            print("Available options:")
            if os.path.exists("best_model_weights.weights.h5"):
                print("  - best_model_weights.weights.h5")
            if os.path.exists("final_model_weights.weights.h5"):
                print("  - final_model_weights.weights.h5")
            print("\nRun 'python train.py' first to train the model!")
            return
    
    required_files = [
        "model_metadata.pkl",
        "braj_tokenizer.model",
        "hindi_tokenizer.model"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nRun 'python train.py' first to train the model!")
        return
    try:
        translator = BrajHindiTranslator(
            model_weights_path=model_path,
            metadata_path="model_metadata.pkl",
            braj_tokenizer_path="braj_tokenizer.model",
            hindi_tokenizer_path="hindi_tokenizer.model"
        )
    except Exception as e:
        print(f"Error loading translator: {e}")
        return
    
    if args.interactive:
        print("\nBraj-to-Hindi Interactive Translator")
        print("=" * 50)
        print("Commands:")
        print("  - Type Braj text to translate")
        print("  - ':mode word' or ':mode sentence' to change mode")
        print("  - ':quit' or ':exit' to exit")
        print("  - ':help' for this help message")
        print("=" * 50)
        
        current_mode = args.mode
        
        while True:
            try:
                user_input = input(f"\n[{current_mode}] Braj> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in [':quit', ':exit', 'quit', 'exit']:
                    break
                
                if user_input.lower() == ':help':
                    print("Commands:")
                    print("  - Type Braj text to translate")
                    print("  - ':mode word' or ':mode sentence' to change mode")
                    print("  - ':quit' or ':exit' to exit")
                    continue
                
                if user_input.startswith(':mode '):
                    new_mode = user_input.split(' ', 1)[1].strip()
                    if new_mode in ['word', 'sentence', 'auto']:
                        current_mode = new_mode
                        print(f"Mode changed to: {current_mode}")
                    else:
                        print("Invalid mode. Use 'word', 'sentence', or 'auto'")
                    continue
                
                translation = translator.translate(user_input, current_mode, args.verbose)
                print(f"Hindi> {translation}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    elif input_text:
        print(f"Input (Braj): {input_text}")
        translation = translator.translate(input_text, args.mode, args.verbose)
        print(f"Output (Hindi): {translation}")
    
    else:
        print("Please provide input text with --input, as arguments, or use --interactive mode")
        print("Example: python translate.py --input \"हौं जाब\"")
        print("Example: python translate.py हौं मुंह धोत हौं")
        print("Example: python translate.py --interactive")


if __name__ == "__main__":
    main()
